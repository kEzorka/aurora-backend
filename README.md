# Aurora Weather Backend

Бэкенд глобального прогноза Aurora (сетка 0.25°, 10 суток) и истории ERA5 с
1940 года. Сервис хранит готовые данные в Zarr v3: API не зависит от GPU и
продолжает отдавать последний опубликованный прогноз, когда inference-узел
выключен.

Актуальные критерии и доказательства: [`docs/PROGRESS.md`](docs/PROGRESS.md),
[`docs/BACKLOG.md`](docs/BACKLOG.md) и [`report.md`](report.md). Архитектурная
точка входа — [`docs/README.md`](docs/README.md).

## Что уже работает

| Контур | Реализация |
|---|---|
| Приём | ECMWF/GFS range-download, ARCO/CDS, два срока analysis, checksum, ретраи и атомарный `analysis/recent` |
| Контракт | 91 поле AuroraV1p5, канонические оси/единицы, четыре уровня валидации |
| Хранилище | Zarr v3, раскладки maps/series, шардинг, атомарная публикация, ротация, месячные ERA5 |
| История | SQLite-кэш, negative cache, LRU, CDS-ряды и ARCO-карты |
| API | forecast/history point и grid, coverage/health, ETag/TTL, лимиты, async ZIP/Zarr export |
| Pipeline | durable inference queue, canonical `aurora.Batch`, RMSE/ACC, baselines и отчёт бюджета цикла |
| Demo | численное значение в точке и GIF из сеточных кадров на `/` и `/demo` |

Внешне заблокированы только шаги, которые локально нельзя честно принять:
GPU rollout `AuroraV1p5` и его новые метрики, xpublish/VirtualiZarr/Icechunk,
живые ECMWF/ERA5 и материализация полного production-архива. Эти зависимости
объявлены в профилях, а `make setup` показывает их состояние; незапущенные
шаги не помечены завершёнными.

## Быстрый запуск

Нужен Python 3.11–3.12 и `uv`.

```bash
make venv
make setup
make lint
make test
make demo       # http://127.0.0.1:8000/
```

`make demo` строит небольшой синтетический прогноз штатными writer/publisher,
затем запускает то же FastAPI-приложение. Mock-ветки в API нет.

Основные операторские команды:

```bash
make ingest-analysis INIT=2026-08-01T00:00:00Z ROOT=/data/aurora
make history-monthly ROOT=/data/aurora FROM=1940-01-01 TO=2026-04-30T23:00Z
make forecast INIT=2026-08-01T00:00:00Z
make metrics FORECAST=... TRUTH=... CLIMATOLOGY=... VAR=2t INIT=...
make timing-report MANIFEST=/data/aurora/runs/.../manifest.json
make serve
```

Service и inference — разные окружения. `requirements/service.txt` не содержит
Torch; `requirements/inference.txt` устанавливается только на CUDA-узле.
Секреты находятся в окружении и `~/.cdsapirc`, никогда в Git/манифестах.

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/v1/forecast/point` | численный ряд прогноза в ближайшем узле |
| GET | `/v1/forecast/grid` | компактная карта на срок |
| GET | `/v1/history/point` | ряд ERA5, `raw6h/daily/monthly` |
| GET | `/v1/history/grid` | карта ERA5 или pinned monthly |
| GET | `/v1/meta/coverage` | фактические слои, сроки, переменные и лимиты |
| GET | `/v1/health` | доступность диска и свежесть прогноза |
| POST | `/v1/export` | durable ZIP/Zarr-задача с TTL и polling |

OpenAPI доступен на `/v1/docs`. Полный контракт, единицы, коды ошибок и тело
export-запроса описаны в [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

## Структура

```text
contracts/    единственный канон полей, осей, единиц и слоёв
adapters/     грязь источников → канонический xarray Dataset
validators/   GRIB, структура, семантика, физика и sanity
pipeline/     расписание, ingest, очередь, Batch, метрики
storage/      Zarr v3, публикация, раскладки и ротация
cache/        origin proxy, SQLite index, LRU и отчёты
api/          FastAPI; читает Store/Cache, модель не импортирует
frontend/     статический demo и браузерный GIF89a encoder
tests/        офлайн-фикстуры и контрактные проверки
```

Направление зависимостей закреплено тестом: API → Store/Cache; адаптеры не
знают о хранилище, а service-контур не импортирует модель.

## Сохранённая история

`archive/server-state` и `docs/notes/previous-version-readme.md` — read-only
контекст прежней реализации. `bench/results/` и `bench/charts/` сохраняют
невоспроизводимые после школы GPU-замеры, но не считаются приёмкой текущего
AuroraV1p5 pipeline.
