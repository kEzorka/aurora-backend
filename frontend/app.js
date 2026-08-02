(function runAuroraDemo() {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const result = $("#result");
  const connection = $("#connection");
  const variables = {
    t2m: { response: "t2m", label: "Температура", fallbackUnit: "degC" },
    wind: { response: "wind_speed", label: "Скорость ветра", fallbackUnit: "m/s" },
    msl: { response: "msl", label: "Давление", fallbackUnit: "hPa" },
  };

  let coverage = null;
  let gifUrl = null;

  function apiUrl(path, params) {
    const url = new URL(path, window.location.origin);
    for (const [name, value] of Object.entries(params)) url.searchParams.set(name, value);
    return url;
  }

  async function request(path, params) {
    const response = await fetch(apiUrl(path, params), { headers: { Accept: "application/json" } });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    return body;
  }

  function parseUtc(value) {
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) throw new Error("Укажите дату и время");
    return date;
  }

  function iso(value) {
    return parseUtc(value).toISOString().replace(".000Z", "Z");
  }

  function localInput(isoDate) {
    const date = new Date(isoDate);
    const local = new Date(date.valueOf() - date.getTimezoneOffset() * 60_000);
    return local.toISOString().slice(0, 16);
  }

  function dateLabel(value) {
    return new Intl.DateTimeFormat("ru-RU", {
      weekday: "short",
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  }

  function setBusy(button, busy, label) {
    button.disabled = busy;
    button.dataset.label ||= button.textContent.trim();
    button.textContent = busy ? label : button.dataset.label;
  }

  function showError(error) {
    result.className = "result error";
    result.innerHTML = `<div class="error-icon">!</div><div><strong>Не удалось получить прогноз</strong><p>${escapeHtml(error.message)}</p></div>`;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[character]);
  }

  function selectTab(name) {
    const point = name === "point";
    $("#point-tab").classList.toggle("active", point);
    $("#animation-tab").classList.toggle("active", !point);
    $("#point-tab").setAttribute("aria-selected", String(point));
    $("#animation-tab").setAttribute("aria-selected", String(!point));
    $("#point-panel").hidden = !point;
    $("#animation-panel").hidden = point;
    $("#point-panel").classList.toggle("hidden", !point);
    $("#animation-panel").classList.toggle("hidden", point);
  }

  async function loadCoverage() {
    try {
      coverage = await request("/v1/meta/coverage", {});
      const layer = coverage.layers.find((item) => item.step_hours === 6) || coverage.layers[0];
      const first = new Date(layer.from);
      const pointTime = new Date(Math.min(first.valueOf() + 24 * 3_600_000, new Date(layer.to).valueOf()));
      const animationEnd = new Date(Math.min(first.valueOf() + 72 * 3_600_000, new Date(layer.to).valueOf()));
      $("#point-time").value = localInput(pointTime);
      $("#animation-from").value = localInput(first);
      $("#animation-to").value = localInput(animationEnd);
      connection.classList.add("online");
      connection.querySelector("span:last-child").textContent = `Прогноз до ${dateLabel(layer.to)}`;
    } catch (error) {
      connection.classList.add("offline");
      connection.querySelector("span:last-child").textContent = "Данные пока недоступны";
      showError(error);
    }
  }

  async function showPoint(event) {
    event.preventDefault();
    const button = $("#point-submit");
    setBusy(button, true, "Получаем…");
    result.className = "result loading";
    result.innerHTML = '<div class="spinner"></div><p>Читаем ближайший узел сетки…</p>';
    try {
      const requested = iso($("#point-time").value);
      const variable = variables[$("#point-var").value];
      const body = await request("/v1/forecast/point", {
        lat: $("#point-lat").value,
        lon: $("#point-lon").value,
        vars: $("#point-var").value,
        from: requested,
        to: requested,
        units: "human",
      });
      const value = body.series[variable.response][0];
      const unit = body.units[variable.response] || variable.fallbackUnit;
      const nearest = body.query.nearest_grid;
      result.className = "result point-result";
      result.innerHTML = `
        <div class="result-copy">
          <p class="result-kicker">${escapeHtml(variable.label)} · ${escapeHtml(dateLabel(body.times[0]))}</p>
          <div class="big-value">${value === null ? "—" : escapeHtml(value)}<span>${escapeHtml(unit)}</span></div>
          <p>Ближайший узел: ${nearest.lat.toFixed(2)}°, ${nearest.lon.toFixed(2)}°</p>
        </div>
        <div class="weather-symbol" aria-hidden="true"><span></span></div>`;
    } catch (error) {
      showError(error);
    } finally {
      setBusy(button, false, "");
    }
  }

  function forecastTimes(from, to) {
    const start = parseUtc(from);
    const end = parseUtc(to);
    if (end < start) throw new Error("Конец периода раньше начала");
    const times = [];
    for (let cursor = start.valueOf(); cursor <= end.valueOf(); cursor += 6 * 3_600_000) {
      times.push(new Date(cursor));
    }
    if (times.length < 2) throw new Error("Для GIF нужен период хотя бы 6 часов");
    if (times.length > 20) throw new Error("Выберите период не более 114 часов (20 кадров)");
    return times;
  }

  function colour(index) {
    const stops = [
      [26, 65, 132], [31, 138, 190], [117, 203, 190],
      [255, 224, 120], [244, 109, 67], [157, 46, 70],
    ];
    const position = (index / 239) * (stops.length - 1);
    const left = Math.min(Math.floor(position), stops.length - 2);
    const fraction = position - left;
    return stops[left].map((value, channel) => Math.round(
      value + (stops[left + 1][channel] - value) * fraction,
    ));
  }

  function palette() {
    const colours = Array.from({ length: 240 }, (_, index) => colour(index));
    colours.push([17, 32, 53]);
    while (colours.length < 256) colours.push([245, 248, 252]);
    return colours;
  }

  function framePixels(values, minimum, maximum) {
    const span = maximum - minimum || 1;
    return Uint8Array.from(values, (value) => {
      if (value === null || !Number.isFinite(value)) return 240;
      return Math.max(0, Math.min(239, Math.round(((value - minimum) / span) * 239)));
    });
  }

  function paintPreview(canvas, pixels, width, height, colours) {
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    const image = context.createImageData(width, height);
    pixels.forEach((index, offset) => {
      const [red, green, blue] = colours[index];
      image.data[offset * 4] = red;
      image.data[offset * 4 + 1] = green;
      image.data[offset * 4 + 2] = blue;
      image.data[offset * 4 + 3] = 255;
    });
    context.putImageData(image, 0, 0);
  }

  async function showAnimation(event) {
    event.preventDefault();
    const button = $("#animation-submit");
    setBusy(button, true, "Собираем кадры…");
    result.className = "result loading";
    result.innerHTML = '<div class="spinner"></div><p id="animation-progress">Подготавливаем запрос…</p>';
    try {
      const times = forecastTimes($("#animation-from").value, $("#animation-to").value);
      const variableName = $("#animation-var").value;
      const frames = [];
      for (let index = 0; index < times.length; index += 1) {
        $("#animation-progress").textContent = `Загружаем кадр ${index + 1} из ${times.length}…`;
        frames.push(await request("/v1/forecast/grid", {
          bbox: $("#animation-bbox").value,
          var: variableName,
          time: times[index].toISOString(),
          stride: "2",
          units: "human",
        }));
      }

      const numbers = frames.flatMap((frame) => frame.values.filter(Number.isFinite));
      if (!numbers.length) throw new Error("В выбранной области нет численных значений");
      const minimum = Math.min(...numbers);
      const maximum = Math.max(...numbers);
      const shape = frames[0].grid.shape;
      const colours = palette();
      const pixels = frames.map((frame) => framePixels(frame.values, minimum, maximum));
      const blob = window.AuroraGif.encode({
        width: shape[1], height: shape[0], palette: colours, frames: pixels, delay: 70,
      });
      if (gifUrl) URL.revokeObjectURL(gifUrl);
      gifUrl = URL.createObjectURL(blob);

      const variable = variables[variableName];
      const unit = frames[0].units[variable.response] || variable.fallbackUnit;
      result.className = "result animation-result";
      result.innerHTML = `
        <div class="animation-preview">
          <canvas id="map-preview" aria-label="Последний кадр прогноза"></canvas>
          <span>${escapeHtml(dateLabel(frames.at(-1).time))}</span>
        </div>
        <div class="animation-copy">
          <p class="result-kicker">${escapeHtml(variable.label)}</p>
          <h3>${frames.length} кадров готовы</h3>
          <p>${minimum.toFixed(1)}…${maximum.toFixed(1)} ${escapeHtml(unit)} · каждые 6 часов</p>
          <a class="download-button" id="gif-download" href="${gifUrl}" download="aurora-forecast.gif">Скачать GIF</a>
        </div>`;
      paintPreview($("#map-preview"), pixels.at(-1), shape[1], shape[0], colours);
    } catch (error) {
      showError(error);
    } finally {
      setBusy(button, false, "");
    }
  }

  $("#point-tab").addEventListener("click", () => selectTab("point"));
  $("#animation-tab").addEventListener("click", () => selectTab("animation"));
  $("#point-panel").addEventListener("submit", showPoint);
  $("#animation-panel").addEventListener("submit", showAnimation);
  loadCoverage();
})();
