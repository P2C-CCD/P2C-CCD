(function () {
  const data = window.P2C_SPACE_DATA;
  const mount = document.getElementById("space-viewer");
  if (!data || !mount) return;

  const frameCount = Array.isArray(data.frames) ? data.frames.length : 0;
  let frameIndex = Math.max(0, Math.min(data.initialFrame || 0, Math.max(frameCount - 1, 0)));
  let playing = false;
  let timer = null;
  let rafHandle = 0;
  let drag = null;

  const state = {
    yaw: data.camera?.yaw ?? 0.75,
    pitch: data.camera?.pitch ?? -0.28,
    radius: data.camera?.radius ?? 6.0,
    minRadius: data.camera?.minRadius ?? 2.0,
    maxRadius: data.camera?.maxRadius ?? 12.0
  };

  mount.innerHTML = `
    <div class="space-root">
      <section class="space-stage">
        <canvas class="space-canvas"></canvas>
        <div class="space-overlay">
          <p class="space-title"></p>
          <p class="space-subtitle"></p>
        </div>
        <div class="space-hint"></div>
        <div class="space-frame"></div>
      </section>
      <section class="space-controls">
        <div class="space-buttons">
          <button class="space-btn" type="button" data-action="prev">Prev</button>
          <button class="space-btn" type="button" data-action="toggle">Play</button>
          <button class="space-btn" type="button" data-action="next">Next</button>
        </div>
        <input class="space-slider" type="range" min="0" max="${Math.max(frameCount - 1, 0)}" step="1" value="${frameIndex}" />
        <p class="space-help"></p>
      </section>
    </div>
  `;

  const root = mount.querySelector(".space-root");
  if (data.accent) {
    root.style.setProperty("--space-accent", data.accent);
  }

  const canvas = mount.querySelector(".space-canvas");
  const stage = mount.querySelector(".space-stage");
  const ctx = canvas.getContext("2d", { alpha: false });
  const titleNode = mount.querySelector(".space-title");
  const subtitleNode = mount.querySelector(".space-subtitle");
  const hintNode = mount.querySelector(".space-hint");
  const frameNode = mount.querySelector(".space-frame");
  const helpNode = mount.querySelector(".space-help");
  const slider = mount.querySelector(".space-slider");
  const playButton = mount.querySelector('[data-action="toggle"]');
  const prevButton = mount.querySelector('[data-action="prev"]');
  const nextButton = mount.querySelector('[data-action="next"]');

  titleNode.textContent = data.title || "Interactive space";
  subtitleNode.textContent = data.subtitle || "";
  hintNode.textContent = data.hint || "Drag to orbit 360 - wheel to zoom - slider to scrub";
  helpNode.textContent = data.help || "Drag inside the window to inspect the 3D space from any angle.";

  const meshes = data.meshes || {};

  function normalizeFrame(frame) {
    for (const surface of frame.surfaces || []) {
      if (Array.isArray(surface.vertices)) surface._vertices = Float32Array.from(surface.vertices);
      if (Array.isArray(surface.triangles)) surface._triangles = Uint32Array.from(surface.triangles);
    }
    if (frame.packedBalls) {
      if (Array.isArray(frame.packedBalls.centers)) frame.packedBalls._centers = Float32Array.from(frame.packedBalls.centers);
      if (Array.isArray(frame.packedBalls.ids)) frame.packedBalls._ids = Uint32Array.from(frame.packedBalls.ids);
    }
    for (const cloud of frame.clouds || []) {
      if (Array.isArray(cloud.points)) cloud._points = Float32Array.from(cloud.points);
      if (Array.isArray(cloud.colors)) cloud._colors = Uint8Array.from(cloud.colors);
    }
  }

  function normalizeMesh(mesh) {
    if (Array.isArray(mesh.vertices)) mesh._vertices = Float32Array.from(mesh.vertices);
    if (Array.isArray(mesh.triangles)) mesh._triangles = Uint32Array.from(mesh.triangles);
  }

  for (const frame of data.frames || []) normalizeFrame(frame);
  for (const mesh of Object.values(meshes)) normalizeMesh(mesh);

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function rgba(color, alpha) {
    return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
  }

  function mixColor(base, lift, t) {
    return base.map((channel, index) =>
      clamp(Math.round(channel * (0.58 + 0.46 * t) + lift[index] * (0.1 + 0.12 * t)), 0, 255)
    );
  }

  function shadeStroke(base, t) {
    return base.map(channel => clamp(Math.round(channel * (0.42 + 0.22 * t)), 0, 255));
  }

  function resizeCanvas() {
    const rect = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    requestRender();
  }

  function requestRender() {
    if (rafHandle) return;
    rafHandle = window.requestAnimationFrame(() => {
      rafHandle = 0;
      render();
    });
  }

  function stopPlayback() {
    playing = false;
    if (timer) {
      window.clearInterval(timer);
      timer = null;
    }
    playButton.textContent = "Play";
  }

  function startPlayback() {
    if (playing || frameCount <= 1) return;
    playing = true;
    playButton.textContent = "Pause";
    timer = window.setInterval(() => {
      frameIndex = (frameIndex + 1) % frameCount;
      slider.value = String(frameIndex);
      requestRender();
    }, data.playbackMs || 850);
  }

  function togglePlayback() {
    if (playing) stopPlayback();
    else startPlayback();
  }

  function setFrame(index) {
    frameIndex = Math.max(0, Math.min(index, frameCount - 1));
    slider.value = String(frameIndex);
    requestRender();
  }

  function rotateOrbit(x, y, z) {
    const cy = Math.cos(state.yaw);
    const sy = Math.sin(state.yaw);
    const cp = Math.cos(state.pitch);
    const sp = Math.sin(state.pitch);

    const x1 = cy * x + sy * z;
    const z1 = -sy * x + cy * z;
    const y1 = cp * y - sp * z1;
    const z2 = sp * y + cp * z1;
    return [x1, y1, z2];
  }

  function project(x, y, z, width, height, focal) {
    const cameraZ = z + state.radius;
    if (cameraZ <= 0.05) return null;
    const px = width * 0.5 + (x * focal) / cameraZ;
    const py = height * 0.55 - (y * focal) / cameraZ;
    return { x: px, y: py, depth: cameraZ };
  }

  function drawGrid(width, height, focal) {
    const grid = data.grid;
    if (!grid) return;
    const size = grid.size ?? 4;
    const divisions = grid.divisions ?? 18;
    const majorEvery = Math.max(1, grid.majorEvery ?? 4);
    const y = grid.y ?? 0;
    const step = (size * 2) / divisions;
    const centerIndex = Math.floor(divisions / 2);
    for (let i = 0; i <= divisions; i += 1) {
      const p = -size + i * step;
      const a = project(...rotateOrbit(-size, y, p), width, height, focal);
      const b = project(...rotateOrbit(size, y, p), width, height, focal);
      const c = project(...rotateOrbit(p, y, -size), width, height, focal);
      const d = project(...rotateOrbit(p, y, size), width, height, focal);
      const offset = Math.abs(i - centerIndex);
      const isAxis = i === centerIndex;
      const isMajor = !isAxis && offset % majorEvery === 0;
      if (a && b) {
        ctx.lineWidth = isAxis ? 1.3 : isMajor ? 1.0 : 0.8;
        ctx.strokeStyle = isAxis ? "rgba(240,132,132,0.48)" : isMajor ? "rgba(193,209,227,0.75)" : "rgba(214,224,238,0.62)";
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      if (c && d) {
        ctx.lineWidth = isAxis ? 1.3 : isMajor ? 1.0 : 0.8;
        ctx.strokeStyle = isAxis ? "rgba(112,164,229,0.48)" : isMajor ? "rgba(193,209,227,0.75)" : "rgba(214,224,238,0.62)";
        ctx.beginPath();
        ctx.moveTo(c.x, c.y);
        ctx.lineTo(d.x, d.y);
        ctx.stroke();
      }
    }
  }

  function collectSurfaceTriangles(frame, width, height, focal) {
    const drawCalls = [];
    const lift = [255, 255, 255];
    const lightDirection = [0.34, 0.48, -0.81];

    for (const surface of frame.surfaces || []) {
      const mesh = surface.mesh ? meshes[surface.mesh] : null;
      const vertices = surface._vertices || mesh?._vertices;
      const triangles = surface._triangles || mesh?._triangles;
      if (!vertices || !triangles) continue;
      const translate = surface.translate || [0, 0, 0];
      const tx = translate[0] || 0;
      const ty = translate[1] || 0;
      const tz = translate[2] || 0;
      const color = surface.color || [74, 163, 255];
      const alpha = surface.alpha ?? 0.76;
      const wireAlpha = surface.wireAlpha ?? 0.14;
      const strokeWidth = surface.strokeWidth ?? 0.75;
      const cull = surface.cull !== false;

      for (let i = 0; i < triangles.length; i += 3) {
        const aIndex = triangles[i] * 3;
        const bIndex = triangles[i + 1] * 3;
        const cIndex = triangles[i + 2] * 3;

        const aWorld = [vertices[aIndex] + tx, vertices[aIndex + 1] + ty, vertices[aIndex + 2] + tz];
        const bWorld = [vertices[bIndex] + tx, vertices[bIndex + 1] + ty, vertices[bIndex + 2] + tz];
        const cWorld = [vertices[cIndex] + tx, vertices[cIndex + 1] + ty, vertices[cIndex + 2] + tz];

        const aOrbit = rotateOrbit(aWorld[0], aWorld[1], aWorld[2]);
        const bOrbit = rotateOrbit(bWorld[0], bWorld[1], bWorld[2]);
        const cOrbit = rotateOrbit(cWorld[0], cWorld[1], cWorld[2]);

        const abx = bOrbit[0] - aOrbit[0];
        const aby = bOrbit[1] - aOrbit[1];
        const abz = bOrbit[2] - aOrbit[2];
        const acx = cOrbit[0] - aOrbit[0];
        const acy = cOrbit[1] - aOrbit[1];
        const acz = cOrbit[2] - aOrbit[2];

        const nx = aby * acz - abz * acy;
        const ny = abz * acx - abx * acz;
        const nz = abx * acy - aby * acx;
        const normalLength = Math.hypot(nx, ny, nz);
        if (normalLength < 1e-6) continue;

        const normal = [nx / normalLength, ny / normalLength, nz / normalLength];
        if (cull && normal[2] >= -0.01) continue;

        const pa = project(aOrbit[0], aOrbit[1], aOrbit[2], width, height, focal);
        const pb = project(bOrbit[0], bOrbit[1], bOrbit[2], width, height, focal);
        const pc = project(cOrbit[0], cOrbit[1], cOrbit[2], width, height, focal);
        if (!pa || !pb || !pc) continue;

        const area = Math.abs((pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x));
        if (area < 0.06) continue;

        const diffuse = Math.max(
          0,
          normal[0] * lightDirection[0] + normal[1] * lightDirection[1] + normal[2] * lightDirection[2]
        );
        const fillColor = mixColor(color, lift, diffuse);
        const strokeColor = shadeStroke(color, diffuse);

        drawCalls.push({
          depth: (pa.depth + pb.depth + pc.depth) / 3,
          alpha,
          wireAlpha,
          strokeWidth,
          fillColor,
          strokeColor,
          points: [pa, pb, pc]
        });
      }
    }

    drawCalls.sort((a, b) => b.depth - a.depth);
    return drawCalls;
  }

  function drawSurfaceTriangles(triangles) {
    for (const triangle of triangles) {
      ctx.beginPath();
      ctx.moveTo(triangle.points[0].x, triangle.points[0].y);
      ctx.lineTo(triangle.points[1].x, triangle.points[1].y);
      ctx.lineTo(triangle.points[2].x, triangle.points[2].y);
      ctx.closePath();
      ctx.fillStyle = rgba(triangle.fillColor, triangle.alpha);
      ctx.fill();
      if (triangle.wireAlpha > 0) {
        ctx.strokeStyle = rgba(triangle.strokeColor, triangle.wireAlpha);
        ctx.lineWidth = triangle.strokeWidth;
        ctx.stroke();
      }
    }
  }

  function collectProjectedSpheres(frame, width, height, focal) {
    const projected = [];
    if (frame.packedBalls) {
      const centers = frame.packedBalls._centers || Float32Array.from(frame.packedBalls.centers || []);
      const ids = frame.packedBalls._ids || null;
      const radii = data.ballRadii || [];
      const waveIds = data.ballWaveIds || [];
      const palette = data.ballPalette || [[85, 167, 235]];
      const alpha = frame.packedBalls.alpha ?? 0.88;
      const strokeAlpha = frame.packedBalls.strokeAlpha ?? 0.24;
      const strokeWidth = frame.packedBalls.strokeWidth ?? 0.75;
      for (let i = 0; i < centers.length; i += 3) {
        const particleId = ids ? ids[i / 3] : i / 3;
        const orbit = rotateOrbit(centers[i] || 0, centers[i + 1] || 0, centers[i + 2] || 0);
        const hit = project(orbit[0], orbit[1], orbit[2], width, height, focal);
        if (!hit) continue;
        const radius = Math.max(1.5, ((radii[particleId] || 0.1) * focal) / hit.depth);
        const wave = waveIds[particleId] || 0;
        projected.push({
          x: hit.x,
          y: hit.y,
          depth: hit.depth,
          radius,
          color: palette[wave % palette.length] || palette[0],
          alpha,
          strokeAlpha,
          strokeWidth
        });
      }
    }
    for (const sphere of frame.balls || []) {
      const center = sphere.center || [0, 0, 0];
      const orbit = rotateOrbit(center[0] || 0, center[1] || 0, center[2] || 0);
      const hit = project(orbit[0], orbit[1], orbit[2], width, height, focal);
      if (!hit) continue;
      const radius = Math.max(1.5, ((sphere.radius || 0.1) * focal) / hit.depth);
      projected.push({
        x: hit.x,
        y: hit.y,
        depth: hit.depth,
        radius,
        color: sphere.color || [85, 167, 235],
        alpha: sphere.alpha ?? 0.9,
        strokeAlpha: sphere.strokeAlpha ?? 0.3,
        strokeWidth: sphere.strokeWidth ?? 1.0
      });
    }
    projected.sort((a, b) => b.depth - a.depth);
    return projected;
  }

  function drawProjectedSpheres(spheres) {
    for (const sphere of spheres) {
      const highlight = sphere.color.map(channel => clamp(Math.round(channel + (255 - channel) * 0.35), 0, 255));
      const shadow = sphere.color.map(channel => clamp(Math.round(channel * 0.72), 0, 255));
      const gradient = ctx.createRadialGradient(
        sphere.x - sphere.radius * 0.28,
        sphere.y - sphere.radius * 0.36,
        Math.max(1, sphere.radius * 0.16),
        sphere.x,
        sphere.y,
        sphere.radius
      );
      gradient.addColorStop(0, rgba(highlight, sphere.alpha));
      gradient.addColorStop(1, rgba(shadow, sphere.alpha));
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(sphere.x, sphere.y, sphere.radius, 0, Math.PI * 2);
      ctx.fill();
      if (sphere.strokeAlpha > 0) {
        ctx.strokeStyle = rgba(shadow, sphere.strokeAlpha);
        ctx.lineWidth = sphere.strokeWidth;
        ctx.stroke();
      }
    }
  }

  function collectProjectedPoints(frame, width, height, focal) {
    const projected = [];
    for (const cloud of frame.clouds || []) {
      const points = cloud._points || Float32Array.from(cloud.points || []);
      const colors = cloud._colors || null;
      const size = cloud.size ?? 2.1;
      const alpha = cloud.alpha ?? 0.82;
      const shape = cloud.shape || "square";
      const baseColor = cloud.color || [74, 163, 255];
      for (let i = 0, j = 0; i < points.length; i += 3, j += 3) {
        const orbit = rotateOrbit(points[i], points[i + 1], points[i + 2]);
        const hit = project(orbit[0], orbit[1], orbit[2], width, height, focal);
        if (!hit) continue;
        const sizePx = Math.max(shape === "circle" ? 2.4 : 1.2, (size * focal) / (hit.depth * 155));
        const color = colors ? [colors[j], colors[j + 1], colors[j + 2]] : baseColor;
        projected.push({
          x: hit.x,
          y: hit.y,
          depth: hit.depth,
          size: sizePx,
          alpha,
          color,
          shape
        });
      }
    }
    projected.sort((a, b) => b.depth - a.depth);
    return projected;
  }

  function drawProjectedPoints(points) {
    for (const point of points) {
      ctx.fillStyle = rgba(point.color, point.alpha);
      if (point.shape === "circle") {
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.size, 0, Math.PI * 2);
        ctx.fill();
      } else {
        const size = point.size;
        ctx.fillRect(point.x - size * 0.5, point.y - size * 0.5, size, size);
      }
    }
  }

  function render() {
    const rect = stage.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const frame = data.frames[frameIndex];
    if (!frame || width <= 0 || height <= 0) return;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = data.background || "#f7f9fd";
    ctx.fillRect(0, 0, width, height);

    const focal = Math.min(width, height) * (data.focalScale || 0.98);
    drawGrid(width, height, focal);

    drawSurfaceTriangles(collectSurfaceTriangles(frame, width, height, focal));
    drawProjectedSpheres(collectProjectedSpheres(frame, width, height, focal));
    drawProjectedPoints(collectProjectedPoints(frame, width, height, focal));

    frameNode.textContent = frame.label || `Frame ${frameIndex + 1} / ${frameCount}`;
  }

  slider.addEventListener("input", () => {
    stopPlayback();
    setFrame(Number(slider.value));
  });

  prevButton.addEventListener("click", () => {
    stopPlayback();
    setFrame((frameIndex - 1 + frameCount) % frameCount);
  });

  nextButton.addEventListener("click", () => {
    stopPlayback();
    setFrame((frameIndex + 1) % frameCount);
  });

  playButton.addEventListener("click", togglePlayback);

  canvas.addEventListener("pointerdown", event => {
    drag = { x: event.clientX, y: event.clientY };
    canvas.classList.add("is-dragging");
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", event => {
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    state.yaw += dx * 0.0085;
    state.pitch = clamp(state.pitch + dy * 0.0065, -1.25, 1.25);
    requestRender();
  });

  function endDrag(event) {
    if (drag && event.pointerId !== undefined && canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    drag = null;
    canvas.classList.remove("is-dragging");
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => {
    if (!drag) canvas.classList.remove("is-dragging");
  });

  canvas.addEventListener(
    "wheel",
    event => {
      event.preventDefault();
      const scale = event.deltaY > 0 ? 1.08 : 0.92;
      state.radius = clamp(state.radius * scale, state.minRadius, state.maxRadius);
      requestRender();
    },
    { passive: false }
  );

  window.addEventListener("keydown", event => {
    if (event.key === "ArrowLeft") {
      stopPlayback();
      setFrame((frameIndex - 1 + frameCount) % frameCount);
    } else if (event.key === "ArrowRight") {
      stopPlayback();
      setFrame((frameIndex + 1) % frameCount);
    } else if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  });

  new ResizeObserver(resizeCanvas).observe(stage);
  resizeCanvas();
})();
