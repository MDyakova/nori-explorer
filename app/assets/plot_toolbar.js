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
  if (!img || noriRegionState.active) {
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
  if (noriRegionState.active) {
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

// --- region selection (drag one or more rectangles to plot each one's protein/lipid
// distribution; regions persist across image switches until "Clear" is clicked) ---

var NORI_REGION_COLORS = ['#0f766e', '#b45309', '#7c3aed', '#be123c', '#0369a1', '#15803d'];

var noriRegionState = {
  active: false, dragging: false, startX: 0, startY: 0, wrap: null,
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
    return { bbox: r.bbox, image: r.imagePath };
  });
  noriSetReactInputValue('viewer-region-input', JSON.stringify(entries));
}

// The dashed box that tracks the mouse while dragging out a new selection.
function noriDraftBoxFor(wrap) {
  var box = wrap.querySelector('.region-draft-box');
  if (!box) {
    box = document.createElement('div');
    box.className = 'region-box region-draft-box';
    wrap.appendChild(box);
  }
  return box;
}

// Creates (or recreates, e.g. when switching back to an image with existing regions)
// the finalized, numbered, percentage-positioned box for one region.
function noriCreateRegionBox(wrap, id, bbox) {
  var color = NORI_REGION_COLORS[(id - 1) % NORI_REGION_COLORS.length];

  var box = document.createElement('div');
  box.className = 'region-box region-final-box';
  box.dataset.regionId = String(id);
  box.style.left = (bbox.x0 * 100) + '%';
  box.style.top = (bbox.y0 * 100) + '%';
  box.style.width = ((bbox.x1 - bbox.x0) * 100) + '%';
  box.style.height = ((bbox.y1 - bbox.y0) * 100) + '%';
  box.style.borderColor = color;
  box.style.background = 'transparent';

  var badge = document.createElement('span');
  badge.className = 'region-box-badge';
  badge.style.background = color;
  badge.textContent = String(id);
  box.appendChild(badge);

  wrap.appendChild(box);
  return box;
}

// Removes all drawn boxes from `wrap` and redraws only the ones belonging to
// `imagePath` (the image currently shown in it) from the held region data.
function noriRedrawRegionBoxes(wrap, imagePath) {
  wrap.querySelectorAll('.region-box').forEach(function (box) { box.remove(); });
  noriRegionState.regions.forEach(function (r) {
    if (r.imagePath === imagePath) {
      noriCreateRegionBox(wrap, r.id, r.bbox);
    }
  });
}

function noriClearRegions(wrap) {
  wrap.querySelectorAll('.region-box').forEach(function (box) { box.remove(); });
  noriRegionState.regions = [];
  noriRegionState.nextId = 1;
}

document.addEventListener('click', function (event) {
  var btn = event.target.closest('#viewer-select-mode-button');
  if (!btn) {
    return;
  }
  noriRegionState.active = !noriRegionState.active;
  btn.classList.toggle('btn-primary', noriRegionState.active);
  btn.classList.toggle('btn-outline', !noriRegionState.active);
  btn.textContent = noriRegionState.active ? 'Selecting… (click to stop)' : 'Select region';

  document.querySelectorAll('.zoom-scroll').forEach(function (scroller) {
    scroller.classList.toggle('region-select-active', noriRegionState.active);
  });
});

document.addEventListener('click', function (event) {
  var btn = event.target.closest('#viewer-clear-regions-button');
  if (!btn) {
    return;
  }
  document.querySelectorAll('.zoom-image-wrap').forEach(noriClearRegions);
  noriPublishRegions();
});

document.addEventListener('mousedown', function (event) {
  if (!noriRegionState.active || event.button !== 0) {
    return;
  }
  var wrap = event.target.closest('.zoom-image-wrap');
  if (!wrap) {
    return;
  }
  event.preventDefault();

  var rect = wrap.getBoundingClientRect();
  noriRegionState.dragging = true;
  noriRegionState.wrap = wrap;
  noriRegionState.startX = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
  noriRegionState.startY = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);

  var box = noriDraftBoxFor(wrap);
  box.style.left = noriRegionState.startX + 'px';
  box.style.top = noriRegionState.startY + 'px';
  box.style.width = '0px';
  box.style.height = '0px';
  box.style.display = 'block';
});

document.addEventListener('mousemove', function (event) {
  if (!noriRegionState.dragging || !noriRegionState.wrap) {
    return;
  }
  var wrap = noriRegionState.wrap;
  var rect = wrap.getBoundingClientRect();
  var curX = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
  var curY = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);

  var left = Math.min(curX, noriRegionState.startX);
  var top = Math.min(curY, noriRegionState.startY);
  var width = Math.abs(curX - noriRegionState.startX);
  var height = Math.abs(curY - noriRegionState.startY);

  var box = noriDraftBoxFor(wrap);
  box.style.left = left + 'px';
  box.style.top = top + 'px';
  box.style.width = width + 'px';
  box.style.height = height + 'px';
});

document.addEventListener('mouseup', function (event) {
  if (!noriRegionState.dragging || !noriRegionState.wrap) {
    return;
  }
  var wrap = noriRegionState.wrap;
  noriRegionState.dragging = false;
  noriRegionState.wrap = null;

  var draft = wrap.querySelector('.region-draft-box');
  var wrapWidth = wrap.clientWidth;
  var wrapHeight = wrap.clientHeight;
  if (!draft || !wrapWidth || !wrapHeight) {
    return;
  }

  var left = parseFloat(draft.style.left) || 0;
  var top = parseFloat(draft.style.top) || 0;
  var width = parseFloat(draft.style.width) || 0;
  var height = parseFloat(draft.style.height) || 0;
  draft.style.display = 'none';

  if (width < 6 || height < 6) {
    return;
  }

  // Store position/size as percentages of the wrap so the box stays aligned
  // with the image at any zoom level without needing to be recomputed.
  var id = noriRegionState.nextId++;
  var bbox = {
    x0: left / wrapWidth,
    y0: top / wrapHeight,
    x1: (left + width) / wrapWidth,
    y1: (top + height) / wrapHeight,
  };
  noriCreateRegionBox(wrap, id, bbox);

  var imgEl = wrap.querySelector('.zoom-image');
  var imagePath = (imgEl && imgEl.dataset.imagePath) || '';
  noriRegionState.regions.push({ id: id, bbox: bbox, imagePath: imagePath });
  noriPublishRegions();
});
