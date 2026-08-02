/* A deliberately small GIF89a encoder for the demo map.
 * It writes one global 256-colour table and emits a clear code before each
 * pixel. The files are a little larger than optimal, but the encoder stays
 * dependency-free and the dictionary can never desynchronise between frames.
 */
(function exposeGifEncoder(global) {
  "use strict";

  const encoder = new TextEncoder();

  function word(value) {
    return [value & 255, (value >> 8) & 255];
  }

  function bytes(text) {
    return Array.from(encoder.encode(text));
  }

  function blocks(payload) {
    const result = [];
    for (let offset = 0; offset < payload.length; offset += 255) {
      const part = payload.slice(offset, offset + 255);
      result.push(part.length, ...part);
    }
    result.push(0);
    return result;
  }

  function literalPixels(pixels) {
    const clear = 256;
    const end = 257;
    const codes = [];
    for (const pixel of pixels) {
      codes.push(clear, pixel);
    }
    codes.push(end);

    const output = [];
    let bucket = 0;
    let bits = 0;
    for (const code of codes) {
      bucket |= code << bits;
      bits += 9;
      while (bits >= 8) {
        output.push(bucket & 255);
        bucket >>= 8;
        bits -= 8;
      }
    }
    if (bits > 0) output.push(bucket & 255);
    return output;
  }

  function encode({ width, height, palette, frames, delay = 70 }) {
    if (!width || !height || !frames.length) throw new Error("GIF: нет кадров");
    if (palette.length !== 256) throw new Error("GIF: палитра должна содержать 256 цветов");

    const stream = [
      ...bytes("GIF89a"),
      ...word(width),
      ...word(height),
      0xf7, 0, 0,
    ];
    for (const [red, green, blue] of palette) stream.push(red, green, blue);

    // NETSCAPE2.0: бесконечный цикл.
    stream.push(0x21, 0xff, 11, ...bytes("NETSCAPE2.0"), 3, 1, 0, 0, 0);

    for (const pixels of frames) {
      if (pixels.length !== width * height) throw new Error("GIF: неверный размер кадра");
      stream.push(
        0x21, 0xf9, 4, 0, ...word(delay), 0, 0,
        0x2c, 0, 0, 0, 0, ...word(width), ...word(height), 0,
        8,
        ...blocks(literalPixels(pixels)),
      );
    }
    stream.push(0x3b);
    return new Blob([new Uint8Array(stream)], { type: "image/gif" });
  }

  global.AuroraGif = Object.freeze({ encode });
})(window);
