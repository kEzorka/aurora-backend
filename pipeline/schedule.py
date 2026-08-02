"""Расписание цикла и ретраи (docs/PIPELINE.md §2 и §3.5).

Здесь только арифметика времени: когда начинать опрос, когда откатываться на
GFS, когда прогон считается пропущенным и сколько ждать между попытками. Ни
сети, ни диска — иначе главное свойство расписания (оно предсказуемо и его
можно проверить на любой момент) пришлось бы доказывать через мок HTTP.

Смысл этих четырёх моментов — в том, что данных **может не быть**. Анализ 00Z
уходит в рассылку в 05:40 UTC, а в Open Data появляется ещё часа через два, и
разброс тут обычный, а не аварийный (§1). Поэтому опрос начинается заметно
позже публикации, идёт с интервалом, и у него есть два разных конца:

* `fallback_at` — данных ECMWF нет, но прогон ещё возможен на GFS. Это
  деградация, а не отказ, и в манифесте она обязана быть видна отдельно от
  прогона, где GFS выбрали намеренно (`storage.manifest`, ключ `degraded`);
* `publish_by` — прогон не опубликован вовсе. Старый прогноз остаётся
  актуальным, а пропуск записывается на диск: «Пропущенный прогон — нормальная
  ситуация, она должна быть видна, а не замаскирована» (§3.5).

Ретраи отделены от расписания цикла: пауза между попытками одной загрузки —
это про один HTTP-запрос, а `fallback_at` — про весь прогон. Смешивать их
нельзя, иначе потолок попыток начнёт зависеть от того, в какой час суток
загрузка началась.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Final, NamedTuple

#: ISO 8601 в UTC — тот же формат, что в манифесте и в плане приёма.
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"

#: Четыре прогона в сутки: 00, 06, 12, 18 UTC (§2).
CYCLE_HOURS: Final = 6

#: Начало опроса, интервал опроса, дедлайн отката, дедлайн публикации — все
#: четыре от `init_time` прогона, а не от «сейчас».
POLL_AFTER: Final = timedelta(hours=7)
POLL_EVERY: Final = timedelta(minutes=5)
FALLBACK_AFTER: Final = timedelta(hours=8, minutes=30)
PUBLISH_AFTER: Final = timedelta(hours=9, minutes=30)

#: Ретраи одной загрузки. Потолок паузы нужен затем же, зачем потолок попыток:
#: без него шестая пауза уедет за полчаса и съест дедлайн отката целиком.
MAX_ATTEMPTS: Final = 5
BASE_DELAY_SEC: Final = 15.0
MAX_DELAY_SEC: Final = 300.0

#: Фазы цикла. Строки, а не Enum: они уезжают в лог и в отметку о пропуске,
#: где всё равно станут строками, и лишний слой перевода только прячет опечатку.
WAIT: Final = "wait"
POLL: Final = "poll"
FALLBACK: Final = "fallback"
MISSED: Final = "missed"


class ScheduleError(ValueError):
    """Отказ расписания: что за поле, что получено, что ожидалось."""

    def __init__(self, field: str, got: object, expected: object) -> None:
        super().__init__(f"{field}: получено {got!r}, ожидалось {expected!r}")
        self.field = field
        self.got = got
        self.expected = expected


class Cycle(NamedTuple):
    """Четыре момента одного прогона.

    Все — абсолютные времена в UTC, а не смещения: смещения пришлось бы
    складывать в каждом месте, где спрашивают «уже пора?», и один пропущенный
    `+` дал бы прогон, который никогда не откатывается на GFS.
    """

    init_time: str
    poll_from: datetime
    fallback_at: datetime
    publish_by: datetime


def cycle(init_time: str) -> Cycle:
    """Расписание прогона на срок `init_time`.

    Срок обязан лежать на сетке прогонов: 00, 06, 12 или 18 UTC. Прогона на
    03Z не существует, и посчитать ему дедлайны значит завести цикл, которого
    никто не запускает и чей пропуск поэтому никогда не всплывёт.
    """
    moment = _parse(init_time)
    if moment.hour % CYCLE_HOURS or moment.minute or moment.second:
        raise ScheduleError("init_time", init_time, f"каждые {CYCLE_HOURS} ч ровно, от 00 UTC")
    return Cycle(
        init_time=init_time,
        poll_from=moment + POLL_AFTER,
        fallback_at=moment + FALLBACK_AFTER,
        publish_by=moment + PUBLISH_AFTER,
    )


def phase(run: Cycle, now: datetime) -> str:
    """Что делать в момент `now`: ждать, опрашивать, откатываться, признать пропуск.

    Границы включаются в **следующую** фазу: ровно в `fallback_at` опрос уже
    закончен. Иначе последний опрос и откат назначаются на одну секунду, и
    какой из них случится первым, решает планировщик, а не расписание.
    """
    moment = _utc(now)
    if moment < run.poll_from:
        return WAIT
    if moment < run.fallback_at:
        return POLL
    if moment < run.publish_by:
        return FALLBACK
    return MISSED


def poll_times(run: Cycle) -> tuple[datetime, ...]:
    """Моменты опроса Open Data: от `poll_from` каждые 5 мин до дедлайна отката.

    Список конечен намеренно. Опрос «пока не появится» на источнике, который
    сегодня может не появиться вовсе, — это цикл, который держит прогон до
    следующего прогона и молча съедает его окно.
    """
    times = []
    moment = run.poll_from
    while moment < run.fallback_at:
        times.append(moment)
        moment += POLL_EVERY
    return tuple(times)


def delays(
    attempts: int = MAX_ATTEMPTS,
    *,
    base_sec: float = BASE_DELAY_SEC,
    cap_sec: float = MAX_DELAY_SEC,
) -> tuple[float, ...]:
    """Паузы между попытками одной загрузки: экспонента с потолком.

    Пауз на одну меньше, чем попыток: после последней ждать уже нечего, и
    лишняя пауза в конце — это потолок попыток, отодвинутый на пять минут без
    единой дополнительной попытки.
    """
    if attempts < 1:
        raise ScheduleError("attempts", attempts, "хотя бы одна попытка")
    if base_sec <= 0 or cap_sec < base_sec:
        raise ScheduleError("base_sec/cap_sec", (base_sec, cap_sec), "0 < base_sec <= cap_sec")
    return tuple(min(base_sec * 2**number, cap_sec) for number in range(attempts - 1))


def latest_init_time(now: datetime) -> str:
    """Последний прогон, чьё окно опроса уже открылось.

    Это ответ на вопрос «какой прогон должен быть сейчас на диске», и он не
    равен «последнему прошедшему сроку»: анализ за 12Z в 13:00 UTC ещё не
    опубликован, и требовать его — считать нормальную задержку источника
    пропуском.
    """
    moment = _utc(now) - POLL_AFTER
    grounded = moment.replace(
        hour=moment.hour - moment.hour % CYCLE_HOURS, minute=0, second=0, microsecond=0
    )
    return grounded.strftime(TIME_FORMAT)


def _parse(init_time: str) -> datetime:
    try:
        return datetime.strptime(init_time, TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError as bad:
        raise ScheduleError("init_time", init_time, TIME_FORMAT) from bad


def _utc(moment: datetime) -> datetime:
    """Наивное время считается UTC, а не локальным.

    Обратное умолчание Python здесь опаснее: сервер в московской зоне сдвинул
    бы все четыре дедлайна на три часа, и заметно это стало бы по пропущенным
    прогонам, а не по ошибке.
    """
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
