"""Индексы GRIB и байтовые диапазоны (BACKLOG 1.7).

Тексты индексов здесь синтетические, но формой — один в один с настоящими:
`.index` ECMWF это строки JSON с `_offset`/`_length`, `.idx` GFS — поля через
двоеточие без длины вовсе. Настоящие индексы в репозитории не лежат по той же
причине, что и файлы прогонов: они привязаны к дате прогона и живут неделю, а
проверяется здесь арифметика над текстом, которая от даты не зависит.
"""

import pytest

from adapters.errors import AdapterError
from adapters.index import Message, Range, parse_ecmwf, parse_gfs, ranges, select

#: Четыре сообщения ECMWF: два приземных подряд, потом дыра, потом два уровня.
ECMWF_INDEX = "\n".join(
    [
        '{"domain":"g","date":"20260801","time":"0000","step":"0","levtype":"sfc",'
        '"param":"2t","_offset":0,"_length":100}',
        '{"domain":"g","date":"20260801","time":"0000","step":"0","levtype":"sfc",'
        '"param":"msl","_offset":100,"_length":100}',
        '{"domain":"g","date":"20260801","time":"0000","step":"0","levtype":"pl",'
        '"levelist":"850","param":"t","_offset":10000,"_length":250}',
        '{"domain":"g","date":"20260801","time":"0000","step":"0","levtype":"pl",'
        '"levelist":"500","param":"z","_offset":10250,"_length":250}',
        "",
    ]
)

#: Три сообщения GFS. Длины в `.idx` нет ни у одного.
GFS_IDX = "\n".join(
    [
        "1:0:d=2026080100:PRMSL:mean sea level:anl:",
        "2:850:d=2026080100:TMP:2 m above ground:anl:",
        "3:1700:d=2026080100:APCP:surface:0-6 hour acc fcst:",
        "",
    ]
)


def test_ecmwf_index_gives_offset_length_and_what_the_message_is() -> None:
    messages = parse_ecmwf(ECMWF_INDEX)

    assert messages[0] == Message(offset=0, length=100, param="2t", level="")
    assert messages[2] == Message(offset=10000, length=250, param="t", level="850")
    assert len(messages) == 4


def test_a_line_that_is_not_json_names_the_line() -> None:
    """`.index`, отданный вместо данных страницей с ошибкой, — обычное дело.
    Отказ обязан сказать, какая строка, иначе искать её в тысяче строк."""
    with pytest.raises(AdapterError, match="index line 2"):
        parse_ecmwf('{"_offset":0,"_length":10}\n<html>503</html>')


def test_an_index_line_without_offsets_is_refused() -> None:
    """Строка без `_offset` — это не сообщение, а описание чего-то другого.
    Пропустить её значит скачать на одно поле меньше и не заметить."""
    with pytest.raises(AdapterError, match="index line 1"):
        parse_ecmwf('{"param":"2t","levtype":"sfc"}')


def test_an_empty_index_is_refused() -> None:
    with pytest.raises(AdapterError, match="index"):
        parse_ecmwf("\n \n")


def test_gfs_idx_takes_the_length_from_the_next_message() -> None:
    """Длины в `.idx` нет: сообщение кончается там, где начинается следующее."""
    messages = parse_gfs(GFS_IDX)

    assert messages[0] == Message(offset=0, length=850, param="PRMSL", level="mean sea level")
    assert messages[1] == Message(offset=850, length=850, param="TMP", level="2 m above ground")


def test_the_last_gfs_message_runs_to_the_end_of_the_file() -> None:
    """`None`, а не ноль: за последним сообщением файл кончается, и ноль дал бы
    пустой `Range` ровно на то сообщение, за которым чаще всего и приходят."""
    last = parse_gfs(GFS_IDX)[-1]

    assert last.length is None
    assert last.end is None
    assert ranges([last]) == (Range(1700, None),)
    assert Range(1700, None).header == "bytes=1700-"


def test_idx_lines_out_of_order_still_get_positive_lengths() -> None:
    """Разность соседей — это длина только в порядке файла, а `.idx` его лишь
    обычно повторяет. Отрицательную длину сервер удовлетворит куском чужого
    сообщения, и cfgrib прочитает его как обрезанный файл."""
    shuffled = "\n".join(
        [
            "2:850:d=2026080100:TMP:2 m above ground:anl:",
            "1:0:d=2026080100:PRMSL:mean sea level:anl:",
        ]
    )

    messages = parse_gfs(shuffled)

    assert [message.offset for message in messages] == [0, 850]
    assert messages[0].length == 850


def test_a_malformed_idx_line_is_refused() -> None:
    with pytest.raises(AdapterError, match="idx line 1"):
        parse_gfs("1:not-a-number:d=2026080100:TMP:2 m above ground:anl:")
    with pytest.raises(AdapterError, match="idx line 1"):
        parse_gfs("1:0:d=2026080100")


def test_select_keeps_what_was_asked_for_by_parameter_and_level() -> None:
    chosen = select(parse_ecmwf(ECMWF_INDEX), [("2t", ""), ("t", "850")])

    assert [(message.param, message.level) for message in chosen] == [("2t", ""), ("t", "850")]


def test_a_field_missing_from_the_index_is_an_error_not_an_empty_list() -> None:
    """Поле, которого в индексе нет, — это либо опечатка в имени, либо файл,
    выложенный наполовину. Молчаливый пропуск превратится в срез без
    переменной, который заметят на сборке батча, а причину будут искать в
    другом месте."""
    with pytest.raises(AdapterError, match="missing"):
        select(parse_ecmwf(ECMWF_INDEX), [("2t", ""), ("q", "850")])


def test_a_level_asked_for_at_the_surface_is_not_the_same_field() -> None:
    """`t` без уровня и `t` на 850 гПа — разные сообщения, и совпадение имени
    параметра тут ничего не значит."""
    with pytest.raises(AdapterError, match="missing"):
        select(parse_ecmwf(ECMWF_INDEX), [("t", "")])


def test_touching_messages_become_one_request() -> None:
    """Два сообщения впритык — это один `Range`: лишний запрос стоит RTT и
    нового шанса на 5xx."""
    chosen = select(parse_ecmwf(ECMWF_INDEX), [("2t", ""), ("msl", "")])

    assert ranges(chosen, gap=0) == (Range(0, 199),)


def test_the_range_is_inclusive_on_both_ends() -> None:
    """`bytes=0-99` — это сто байт. Забытая единица приклеивает к сообщению
    первый байт следующего, и cfgrib читает его как обрезанный файл."""
    assert ranges([Message(0, 100, "2t")]) == (Range(0, 99),)
    assert Range(0, 99).header == "bytes=0-99"


def test_a_hole_wider_than_the_gap_splits_the_request() -> None:
    """Иначе за двумя полями по краям файла приедет весь файл."""
    chosen = select(parse_ecmwf(ECMWF_INDEX), [("2t", ""), ("t", "850")])

    assert ranges(chosen, gap=0) == (Range(0, 99), Range(10000, 10249))
    # Дыра в 9900 байт дешевле второго запроса, и умолчание её сшивает.
    assert ranges(chosen) == (Range(0, 10249),)


def test_a_message_without_an_end_swallows_everything_after_it() -> None:
    """У «до конца файла» правой границы нет, и любое продолжение уже внутри."""
    messages = [Message(0, None, "APCP"), Message(500, 100, "TMP")]

    assert ranges(messages, gap=0) == (Range(0, None),)


def test_nothing_to_fetch_is_no_requests_rather_than_one_empty() -> None:
    assert ranges([]) == ()


def test_a_negative_gap_is_refused() -> None:
    with pytest.raises(AdapterError, match="gap"):
        ranges([Message(0, 100, "2t")], gap=-1)
