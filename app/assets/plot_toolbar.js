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

  var filename = (btn.getAttribute('data-filename') || 'plot') + '.png';

  if (btn.classList.contains('plot-download-btn')) {
    noriDownloadImage(img.src, filename);
  } else {
    noriCopyImage(img.src, btn);
  }
});
