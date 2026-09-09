function noriDownloadImage(dataUri, filename) {
  var a = document.createElement('a');
  a.href = dataUri;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function noriFlashButton(btn, symbol) {
  var original = btn.textContent;
  btn.textContent = symbol;
  setTimeout(function () { btn.textContent = original; }, 1200);
}

function noriCopyImage(dataUri, btn) {
  fetch(dataUri)
    .then(function (res) { return res.blob(); })
    .then(function (blob) {
      return navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
    })
    .then(function () { noriFlashButton(btn, '✓'); })
    .catch(function () { noriFlashButton(btn, '✗'); });
}

function noriCopyCanvas(canvas, btn) {
  canvas.toBlob(function (blob) {
    if (!blob) {
      noriFlashButton(btn, '✗');
      return;
    }
    navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
      .then(function () { noriFlashButton(btn, '✓'); })
      .catch(function () { noriFlashButton(btn, '✗'); });
  });
}

// For a zoomed/scrolled .zoom-image, render only the crop currently visible in its
// .zoom-scroll viewport (at the size it's currently displayed at) onto a canvas.
function noriVisibleCropCanvas(img) {
  var scroller = img.closest('.zoom-scroll');
  var displayedWidth = img.clientWidth;
  var displayedHeight = img.clientHeight;
  if (!scroller || !displayedWidth || !displayedHeight || !img.naturalWidth) {
    return null;
  }

  var scaleToNatural = img.naturalWidth / displayedWidth;

  var visibleLeft = scroller.scrollLeft;
  var visibleTop = scroller.scrollTop;
  var visibleWidth = Math.min(scroller.clientWidth, displayedWidth - visibleLeft);
  var visibleHeight = Math.min(scroller.clientHeight, displayedHeight - visibleTop);
  if (visibleWidth <= 0 || visibleHeight <= 0) {
    return null;
  }

  var canvas = document.createElement('canvas');
  canvas.width = Math.round(visibleWidth);
  canvas.height = Math.round(visibleHeight);
  canvas.getContext('2d').drawImage(
    img,
    visibleLeft * scaleToNatural, visibleTop * scaleToNatural,
    visibleWidth * scaleToNatural, visibleHeight * scaleToNatural,
    0, 0, canvas.width, canvas.height,
  );
  return canvas;
}

document.addEventListener('click', function (event) {
  var btn = event.target.closest('.plot-copy-btn, .plot-download-btn');
  if (!btn) {
    return;
  }
  var wrap = btn.closest('.plot-wrap');
  var img = wrap ? wrap.querySelector('img') : null;
  if (!img || !img.src || img.src.indexOf('data:') !== 0) {
    return;
  }

  var isDownload = btn.classList.contains('plot-download-btn');
  var baseFilename = btn.getAttribute('data-filename') || 'plot';

  if (img.classList.contains('zoom-image')) {
    var canvas = noriVisibleCropCanvas(img);
    if (!canvas) {
      noriFlashButton(btn, '✗');
      return;
    }
    var zoomFilename = baseFilename + '_' + noriCurrentZoom(img) + 'pct.png';
    if (isDownload) {
      noriDownloadImage(canvas.toDataURL('image/png'), zoomFilename);
    } else {
      noriCopyCanvas(canvas, btn);
    }
    return;
  }

  var filename = baseFilename + '.png';
  if (isDownload) {
    noriDownloadImage(img.src, filename);
  } else {
    noriCopyImage(img.src, btn);
  }
});

// --- click-to-zoom image viewer (img.zoom-image inside div.zoom-scroll) --

var NORI_ZOOM_STEP = 10;   // percentage points per click
var NORI_ZOOM_MIN = 10;    // percent
var NORI_ZOOM_MAX = 800;   // percent

function noriZoomReadout(img) {
  var card = img.closest('.card');
  return card ? card.querySelector('#viewer-zoom-readout') : null;
}

// Hides only the in-progress drag box (finalized regions are positioned in
// percentages, so they stay correctly aligned across zoom changes).
function noriHideRegionBox(img) {
  var wrap = img.closest('.zoom-image-wrap');
  var box = wrap ? wrap.querySelector('.region-draft-box') : null;
  if (box) {
    box.style.display = 'none';
  }
}

function noriApplyZoom(img, pct, clientX, clientY) {
  var naturalWidth = img.naturalWidth;
  if (!naturalWidth) {
    return;
  }
  noriHideRegionBox(img);
  var scroller = img.closest('.zoom-scroll');
  var beforeRect = img.getBoundingClientRect();

  // Fraction of the image under the cursor, so we can re-center on it after resizing.
  var relX = scroller && clientX != null ? (clientX - beforeRect.left) / beforeRect.width : 0.5;
  var relY = scroller && clientY != null ? (clientY - beforeRect.top) / beforeRect.height : 0.5;
  relX = Math.min(Math.max(relX, 0), 1);
  relY = Math.min(Math.max(relY, 0), 1);

  var newWidth = naturalWidth * (pct / 100);
  img.style.width = newWidth + 'px';
  img.dataset.zoomPct = String(pct);

  var readout = noriZoomReadout(img);
  if (readout) {
    readout.textContent = 'Zoom: ' + pct + '%';
  }

  if (scroller) {
    // Wait for layout to pick up the new width before recentring the scroll position.
    requestAnimationFrame(function () {
      var newHeight = img.getBoundingClientRect().height;
      scroller.scrollLeft = relX * newWidth - scroller.clientWidth / 2;
      scroller.scrollTop = relY * newHeight - scroller.clientHeight / 2;
    });
  }
}

function noriCurrentZoom(img) {
  return parseFloat(img.dataset.zoomPct || '100') || 100;
}

// Zoom percentage that fits the whole (natural-size) image inside its .zoom-scroll
// viewport — scales down for images bigger than the visible area, but never scales up
// past 100% (native resolution) for images that are already smaller than it.
function noriFitZoomPct(img) {
  var scroller = img.closest('.zoom-scroll');
  var naturalWidth = img.naturalWidth;
  var naturalHeight = img.naturalHeight;
  if (!scroller || !naturalWidth || !naturalHeight) {
    return 100;
  }
  var fitScale = Math.min(
    scroller.clientWidth / naturalWidth,
    scroller.clientHeight / naturalHeight,
  );
  var pct = Math.round(Math.min(fitScale * 100, 100));
  return Math.min(Math.max(pct, NORI_ZOOM_MIN), NORI_ZOOM_MAX);
}

document.addEventListener('click', function (event) {
  var img = event.target.closest('.zoom-image');
  if (!img || noriRegionState.mode || noriInspectMode) {
    return;
  }
  var pct = Math.min(noriCurrentZoom(img) + NORI_ZOOM_STEP, NORI_ZOOM_MAX);
  noriApplyZoom(img, pct, event.clientX, event.clientY);
});

document.addEventListener('contextmenu', function (event) {
  var img = event.target.closest('.zoom-image');
  if (!img) {
    return;
  }
  event.preventDefault();
  if (noriRegionState.mode) {
    return;
  }
  var pct = Math.max(noriCurrentZoom(img) - NORI_ZOOM_STEP, NORI_ZOOM_MIN);
  noriApplyZoom(img, pct, event.clientX, event.clientY);
});

// Fit the image to its .zoom-scroll viewport whenever a new image is loaded into a
// .zoom-image element. Selected regions are NOT cleared here — they're kept across
// image switches (so you can pick regions from several images and compare them
// together) — only the visual boxes for the now-hidden image are removed; boxes for
// the newly-shown image (if any were drawn on it before) are redrawn from the still-
// held region data.
document.addEventListener('load', function (event) {
  var img = event.target;
  if (!img.classList || !img.classList.contains('zoom-image')) {
    return;
  }
  // Clear any leftover width from a previously displayed image first, so the fit
  // calculation below measures this image's own natural aspect ratio.
  img.style.width = '';
  noriApplyZoom(img, noriFitZoomPct(img), null, null);

  var wrap = img.closest('.zoom-image-wrap');
  if (wrap) {
    noriRedrawRegionBoxes(wrap, img.dataset.imagePath || '');
  }
}, true);

// --- region selection (draw one or more rectangles or polygons to plot each one's
// protein/lipid distribution; regions persist across image switches until "Clear" is
// clicked) ---

var NORI_REGION_COLORS = ['#0f766e', '#b45309', '#7c3aed', '#be123c', '#0369a1', '#15803d'];
var NORI_SVG_NS = 'http://www.w3.org/2000/svg';
var NORI_POLYGON_CLOSE_PX = 8; // click-near-first-vertex radius that closes a polygon

var noriRegionState = {
  mode: null,              // null | 'rect' | 'polygon'
  dragging: false, startPoint: null, wrap: null,   // rectangle drag state
  polyWrap: null, polyPoints: [],                  // polygon-in-progress state
  regions: [], nextId: 1,
};

function noriSetReactInputValue(id, value) {
  var el = document.getElementById(id);
  if (!el) {
    return;
  }
  var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(el, value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

function noriPublishRegions() {
  var entries = noriRegionState.regions.map(function (r) {
    return { type: r.type, points: r.points, image: r.imagePath };
  });
  noriSetReactInputValue('viewer-region-input', JSON.stringify(entries));
}

// Fractional (0..1) position of a client point within `wrap`, clamped to its bounds.
function noriWrapFraction(wrap, clientX, clientY) {
  var rect = wrap.getBoundingClientRect();
  var x = rect.width ? Math.min(Math.max(clientX - rect.left, 0), rect.width) / rect.width : 0;
  var y = rect.height ? Math.min(Math.max(clientY - rect.top, 0), rect.height) / rect.height : 0;
  return { x: x, y: y };
}

function noriPointsAttr(points) {
  return points.map(function (p) { return (p.x * 100) + ',' + (p.y * 100); }).join(' ');
}

function noriRectCorners(points) {
  var p0 = points[0], p1 = points[1];
  var minX = Math.min(p0.x, p1.x), maxX = Math.max(p0.x, p1.x);
  var minY = Math.min(p0.y, p1.y), maxY = Math.max(p0.y, p1.y);
  return [
    { x: minX, y: minY }, { x: maxX, y: minY }, { x: maxX, y: maxY }, { x: minX, y: maxY },
  ];
}

function noriOverlaySvgFor(wrap) {
  var svg = wrap.querySelector('.region-overlay-svg');
  if (!svg) {
    svg = document.createElementNS(NORI_SVG_NS, 'svg');
    svg.setAttribute('class', 'region-overlay-svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    wrap.appendChild(svg);
  }
  return svg;
}

// --- draft (in-progress) shape while dragging a rectangle or placing polygon vertices ---

function noriClearDraftShape(wrap) {
  var svg = wrap.querySelector('.region-overlay-svg');
  if (svg) {
    svg.querySelectorAll('.region-draft-shape').forEach(function (el) { el.remove(); });
  }
  wrap.querySelectorAll('.region-draft-vertex').forEach(function (el) { el.remove(); });
}

function noriUpdateDraftRect(wrap, p0, p1) {
  var svg = noriOverlaySvgFor(wrap);
  var rect = svg.querySelector('.region-draft-shape');
  if (!rect) {
    rect = document.createElementNS(NORI_SVG_NS, 'rect');
    rect.setAttribute('class', 'region-shape region-draft-shape');
    svg.appendChild(rect);
  }
  rect.setAttribute('x', Math.min(p0.x, p1.x) * 100);
  rect.setAttribute('y', Math.min(p0.y, p1.y) * 100);
  rect.setAttribute('width', Math.abs(p1.x - p0.x) * 100);
  rect.setAttribute('height', Math.abs(p1.y - p0.y) * 100);
}

function noriUpdatePolygonDraft(wrap, points, cursor) {
  var svg = noriOverlaySvgFor(wrap);
  var previewPoints = cursor ? points.concat([cursor]) : points;

  var poly = svg.querySelector('.region-draft-shape');
  if (!poly) {
    poly = document.createElementNS(NORI_SVG_NS, 'polyline');
    poly.setAttribute('class', 'region-shape region-draft-shape');
    svg.appendChild(poly);
  }
  poly.setAttribute('points', noriPointsAttr(previewPoints));

  // Plain HTML dots (not SVG shapes in the percentage-scaled viewBox) so their size
  // stays constant on screen at any zoom level, instead of growing with the image.
  wrap.querySelectorAll('.region-draft-vertex').forEach(function (el) { el.remove(); });
  points.forEach(function (p, i) {
    var dot = document.createElement('div');
    dot.className = 'region-draft-vertex' + (i === 0 ? ' region-draft-vertex-first' : '');
    dot.style.left = (p.x * 100) + '%';
    dot.style.top = (p.y * 100) + '%';
    wrap.appendChild(dot);
  });
}

// --- finalized (saved) regions ---

// Creates (or recreates, e.g. when switching back to an image with existing regions)
// the finalized, numbered, percentage-positioned shape for one region.
function noriCreateRegionShape(wrap, id, region) {
  var svg = noriOverlaySvgFor(wrap);
  var color = NORI_REGION_COLORS[(id - 1) % NORI_REGION_COLORS.length];
  var points = region.type === 'polygon' ? region.points : noriRectCorners(region.points);

  var shape = document.createElementNS(NORI_SVG_NS, 'polygon');
  shape.setAttribute('class', 'region-shape region-final-shape');
  shape.setAttribute('points', noriPointsAttr(points));
  shape.setAttribute('data-region-id', String(id));
  shape.style.stroke = color;
  svg.appendChild(shape);

  var minX = Math.min.apply(null, points.map(function (p) { return p.x; }));
  var minY = Math.min.apply(null, points.map(function (p) { return p.y; }));
  var badge = document.createElement('span');
  badge.className = 'region-badge';
  badge.dataset.regionId = String(id);
  badge.style.left = (minX * 100) + '%';
  badge.style.top = (minY * 100) + '%';
  badge.style.background = color;
  badge.textContent = String(id);
  wrap.appendChild(badge);
}

// Removes all drawn shapes/badges from `wrap` and redraws only the ones belonging to
// `imagePath` (the image currently shown in it) from the held region data.
function noriRedrawRegionBoxes(wrap, imagePath) {
  wrap.querySelectorAll('.region-final-shape, .region-badge').forEach(function (el) { el.remove(); });
  noriRegionState.regions.forEach(function (r) {
    if (r.imagePath === imagePath) {
      noriCreateRegionShape(wrap, r.id, r);
    }
  });
}

function noriClearRegions(wrap) {
  wrap.querySelectorAll('.region-final-shape, .region-badge').forEach(function (el) { el.remove(); });
  noriClearDraftShape(wrap);
  noriRegionState.regions = [];
  noriRegionState.nextId = 1;
  if (noriRegionState.polyWrap === wrap) {
    noriRegionState.polyPoints = [];
    noriRegionState.polyWrap = null;
  }
}

// Removes one region (by its position in noriRegionState.regions, which matches the
// order region cards are rendered in) and redraws boxes for whichever image is
// currently shown in each viewer.
function noriDeleteRegion(index) {
  if (index < 0 || index >= noriRegionState.regions.length) {
    return;
  }
  noriRegionState.regions.splice(index, 1);
  document.querySelectorAll('.zoom-image-wrap').forEach(function (wrap) {
    var imgEl = wrap.querySelector('.zoom-image');
    noriRedrawRegionBoxes(wrap, (imgEl && imgEl.dataset.imagePath) || '');
  });
  noriPublishRegions();
}

function noriFinishPolygon(wrap) {
  var points = noriRegionState.polyPoints;
  noriClearDraftShape(wrap);
  noriRegionState.polyPoints = [];
  noriRegionState.polyWrap = null;

  if (points.length < 3) {
    return;
  }
  var id = noriRegionState.nextId++;
  var imgEl = wrap.querySelector('.zoom-image');
  var region = { id: id, type: 'polygon', points: points, imagePath: (imgEl && imgEl.dataset.imagePath) || '' };
  noriCreateRegionShape(wrap, id, region);
  noriRegionState.regions.push(region);
  noriPublishRegions();
}

function noriSetRegionMode(mode) {
  // Cancel any in-progress polygon draft when switching modes.
  if (noriRegionState.polyWrap) {
    noriClearDraftShape(noriRegionState.polyWrap);
  }
  noriRegionState.polyPoints = [];
  noriRegionState.polyWrap = null;
  noriRegionState.dragging = false;
  noriRegionState.wrap = null;

  noriRegionState.mode = noriRegionState.mode === mode ? null : mode;

  var rectBtn = document.getElementById('viewer-select-rect-button');
  var polyBtn = document.getElementById('viewer-select-polygon-button');
  if (rectBtn) {
    var rectActive = noriRegionState.mode === 'rect';
    rectBtn.classList.toggle('btn-primary', rectActive);
    rectBtn.classList.toggle('btn-outline', !rectActive);
    rectBtn.textContent = rectActive ? 'Rectangle (click to stop)' : 'Rectangle';
  }
  if (polyBtn) {
    var polyActive = noriRegionState.mode === 'polygon';
    polyBtn.classList.toggle('btn-primary', polyActive);
    polyBtn.classList.toggle('btn-outline', !polyActive);
    polyBtn.textContent = polyActive ? 'Polygon (dbl-click to finish)' : 'Polygon';
  }

  document.querySelectorAll('.zoom-scroll').forEach(function (scroller) {
    scroller.classList.toggle('region-select-active', !!noriRegionState.mode);
  });
}

document.addEventListener('click', function (event) {
  if (event.target.closest('#viewer-select-rect-button')) {
    noriSetRegionMode('rect');
  } else if (event.target.closest('#viewer-select-polygon-button')) {
    noriSetRegionMode('polygon');
  }
});

document.addEventListener('click', function (event) {
  var btn = event.target.closest('#viewer-clear-regions-button');
  if (!btn) {
    return;
  }
  document.querySelectorAll('.zoom-image-wrap').forEach(noriClearRegions);
  noriPublishRegions();
});

document.addEventListener('click', function (event) {
  var btn = event.target.closest('.region-delete-btn');
  if (!btn) {
    return;
  }
  noriDeleteRegion(parseInt(btn.getAttribute('data-region-index'), 10));
});

document.addEventListener('keydown', function (event) {
  if (event.key !== 'Escape' || !noriRegionState.polyWrap) {
    return;
  }
  noriClearDraftShape(noriRegionState.polyWrap);
  noriRegionState.polyPoints = [];
  noriRegionState.polyWrap = null;
});

// Rectangle: drag out a box.
document.addEventListener('mousedown', function (event) {
  if (!noriRegionState.mode || event.button !== 0) {
    return;
  }
  var wrap = event.target.closest('.zoom-image-wrap');
  if (!wrap) {
    return;
  }
  event.preventDefault();

  if (noriRegionState.mode !== 'rect') {
    return; // polygon vertices are placed on click, not drag
  }
  noriRegionState.dragging = true;
  noriRegionState.wrap = wrap;
  noriRegionState.startPoint = noriWrapFraction(wrap, event.clientX, event.clientY);
  noriUpdateDraftRect(wrap, noriRegionState.startPoint, noriRegionState.startPoint);
});

document.addEventListener('mousemove', function (event) {
  if (!noriRegionState.dragging || !noriRegionState.wrap) {
    return;
  }
  var wrap = noriRegionState.wrap;
  noriUpdateDraftRect(wrap, noriRegionState.startPoint, noriWrapFraction(wrap, event.clientX, event.clientY));
});

document.addEventListener('mouseup', function (event) {
  if (!noriRegionState.dragging || !noriRegionState.wrap) {
    return;
  }
  var wrap = noriRegionState.wrap;
  noriRegionState.dragging = false;
  noriRegionState.wrap = null;
  noriClearDraftShape(wrap);

  var start = noriRegionState.startPoint;
  var end = noriWrapFraction(wrap, event.clientX, event.clientY);
  var wrapRect = wrap.getBoundingClientRect();
  var minFracX = wrapRect.width ? 6 / wrapRect.width : 0;
  var minFracY = wrapRect.height ? 6 / wrapRect.height : 0;
  if (Math.abs(end.x - start.x) < minFracX || Math.abs(end.y - start.y) < minFracY) {
    return;
  }

  var id = noriRegionState.nextId++;
  var imgEl = wrap.querySelector('.zoom-image');
  var region = { id: id, type: 'rect', points: [start, end], imagePath: (imgEl && imgEl.dataset.imagePath) || '' };
  noriCreateRegionShape(wrap, id, region);
  noriRegionState.regions.push(region);
  noriPublishRegions();
});

// Polygon: click to place each vertex; preview the pending edge as the mouse moves.
document.addEventListener('mousemove', function (event) {
  if (noriRegionState.mode !== 'polygon' || !noriRegionState.polyWrap || !noriRegionState.polyPoints.length) {
    return;
  }
  var wrap = noriRegionState.polyWrap;
  noriUpdatePolygonDraft(wrap, noriRegionState.polyPoints, noriWrapFraction(wrap, event.clientX, event.clientY));
});

document.addEventListener('click', function (event) {
  if (noriRegionState.mode !== 'polygon') {
    return;
  }
  var wrap = event.target.closest('.zoom-image-wrap');
  if (!wrap) {
    return;
  }

  if (noriRegionState.polyWrap && noriRegionState.polyWrap !== wrap) {
    noriClearDraftShape(noriRegionState.polyWrap);
    noriRegionState.polyPoints = [];
  }
  noriRegionState.polyWrap = wrap;

  var pt = noriWrapFraction(wrap, event.clientX, event.clientY);

  // Clicking near the first vertex closes the polygon.
  if (noriRegionState.polyPoints.length >= 3) {
    var first = noriRegionState.polyPoints[0];
    var wrapRect = wrap.getBoundingClientRect();
    var dx = (pt.x - first.x) * wrapRect.width;
    var dy = (pt.y - first.y) * wrapRect.height;
    if (Math.sqrt(dx * dx + dy * dy) <= NORI_POLYGON_CLOSE_PX) {
      noriFinishPolygon(wrap);
      return;
    }
  }

  noriRegionState.polyPoints.push(pt);
  noriUpdatePolygonDraft(wrap, noriRegionState.polyPoints, pt);
});

document.addEventListener('dblclick', function (event) {
  if (noriRegionState.mode !== 'polygon' || !noriRegionState.polyWrap) {
    return;
  }
  var wrap = event.target.closest('.zoom-image-wrap');
  if (wrap !== noriRegionState.polyWrap) {
    return;
  }
  event.preventDefault();
  // A dblclick is preceded by two 'click' events, which already placed two (near-
  // duplicate) vertices at this spot — drop the extra one before finalizing.
  if (noriRegionState.polyPoints.length > 3) {
    noriRegionState.polyPoints.pop();
  }
  noriFinishPolygon(wrap);
});

// --- whole-image overlay: click-to-inspect a tile's heatmap(s) ---
// When active, a click on the overlay's .zoom-image reports its fractional (0..1)
// position (via the same noriWrapFraction used for region selection) instead of
// zooming, so the Python callback can map it back to a tile and load its heatmaps.

var noriInspectMode = false;

document.addEventListener('click', function (event) {
  if (event.target.closest('#overlay-inspect-button')) {
    noriInspectMode = !noriInspectMode;
    var btn = document.getElementById('overlay-inspect-button');
    btn.classList.toggle('btn-primary', noriInspectMode);
    btn.classList.toggle('btn-outline', !noriInspectMode);
    btn.textContent = noriInspectMode ? 'Inspect tile (click to stop)' : 'Inspect tile';
    document.querySelectorAll('.zoom-scroll').forEach(function (scroller) {
      scroller.classList.toggle('region-select-active', noriInspectMode);
    });
  }
});

document.addEventListener('click', function (event) {
  if (!noriInspectMode) {
    return;
  }
  var img = event.target.closest('.zoom-image');
  var wrap = img ? img.closest('.zoom-image-wrap') : null;
  if (!wrap) {
    return;
  }
  var frac = noriWrapFraction(wrap, event.clientX, event.clientY);
  noriSetReactInputValue('overlay-tile-click-input', JSON.stringify({ x: frac.x, y: frac.y, t: Date.now() }));
});

// --- All model plots: Prev/Next navigation between per-model cards (all of them are
// rendered on page load; only one .model-card is shown at a time) ---

function noriAllModelsCurrentIndex(cards) {
  for (var i = 0; i < cards.length; i++) {
    if (cards[i].style.display !== 'none') {
      return i;
    }
  }
  return 0;
}

function noriAllModelsShow(index) {
  var cards = Array.prototype.slice.call(document.querySelectorAll('.model-card'));
  var n = cards.length;
  if (!n) {
    return;
  }
  index = ((index % n) + n) % n;
  cards.forEach(function (card, i) {
    card.style.display = i === index ? 'block' : 'none';
  });
  var label = document.getElementById('all-models-nav-label');
  if (label) {
    var name = cards[index].getAttribute('data-model-name') || '';
    label.textContent = name + '  (' + (index + 1) + ' / ' + n + ')';
  }
}

document.addEventListener('click', function (event) {
  var isPrev = event.target.closest('#all-models-prev-button');
  var isNext = event.target.closest('#all-models-next-button');
  if (!isPrev && !isNext) {
    return;
  }
  var cards = Array.prototype.slice.call(document.querySelectorAll('.model-card'));
  var index = noriAllModelsCurrentIndex(cards);
  noriAllModelsShow(index + (isNext ? 1 : -1));
});
