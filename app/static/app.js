const CLASS_LABELS = {
  D00: 'Grieta longitudinal', D10: 'Grieta transversal',
  D20: 'Piel de cocodrilo', D40: 'Bache'
};
const form = document.querySelector('#predict-form');
const input = document.querySelector('#file-input');
const dropzone = document.querySelector('#dropzone');
const fileName = document.querySelector('#file-name');
const confidence = document.querySelector('#confidence');
const confidenceValue = document.querySelector('#confidence-value');
const button = document.querySelector('#submit-button');
const message = document.querySelector('#form-message');
const details = document.querySelector('#details');
const resultsCard = document.querySelector('#results-card');
const summaryGrid = document.querySelector('#summary-grid');
const mediaWrap = document.querySelector('#media-wrap');
const reportLink = document.querySelector('#download-report');
const trackInput = document.querySelector('#track-input');
const recordedAt = document.querySelector('#recorded-at');
const locationFields = document.querySelector('#video-location-fields');
const locationResult = document.querySelector('#location-result');
let timer = null;

confidence.addEventListener('input', () => {
  confidenceValue.value = Number(confidence.value).toFixed(2);
  confidenceValue.textContent = confidenceValue.value;
});
input.addEventListener('change', () => showFile(input.files[0]));
['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault(); dropzone.classList.add('drag');
}));
['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', event => {
  const file = event.dataTransfer.files[0];
  event.preventDefault(); dropzone.classList.remove('drag');
  if (file) { input.files = event.dataTransfer.files; showFile(file); }
});

function showFile(file) {
  if (!file) return;
  const isVideo = /\.(mp4|mov|avi|mkv)$/i.test(file.name);
  locationFields.classList.toggle('hidden', !isVideo);
  if (!isVideo) { trackInput.value = ''; recordedAt.value = ''; }
  fileName.textContent = `${file.name} · ${(file.size / 1048576).toFixed(2)} MB`;
  fileName.classList.remove('hidden');
}
function setMessage(text, error = false) {
  message.textContent = text;
  message.classList.toggle('error', error);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}
function processingView(title, note) {
  resultsCard.className = 'results-card';
  resultsCard.innerHTML = `<div class="processing-view" role="status" aria-live="polite">
    <span class="processing-icon">RG</span><p class="eyebrow">ANÁLISIS EN CURSO</p>
    <h2 id="progress-title"></h2><p id="progress-note"></p>
    <div class="progress-track indeterminate" role="progressbar" aria-label="Progreso del análisis"><div class="progress-fill"></div></div>
    <div class="progress-line"><span id="progress-detail">Preparando archivo</span><strong id="progress-value">…</strong></div>
    <p class="processing-note">Puedes dejar esta pestaña abierta. Un video largo puede tardar varios minutos.</p>
  </div>`;
  updateProgress(title, note, null, 'Preparando archivo');
}
function updateProgress(title, note, percent, detail) {
  document.querySelector('#progress-title').textContent = title;
  document.querySelector('#progress-note').textContent = note;
  document.querySelector('#progress-detail').textContent = detail;
  const track = resultsCard.querySelector('.progress-track');
  const value = document.querySelector('#progress-value');
  if (percent === null) {
    track.classList.add('indeterminate');
    track.removeAttribute('aria-valuenow');
    value.textContent = '…';
  } else {
    track.classList.remove('indeterminate');
    track.setAttribute('aria-valuenow', String(percent));
    track.querySelector('.progress-fill').style.width = `${percent}%`;
    value.textContent = `${percent}%`;
  }
}
function upload(file) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const params = new URLSearchParams({ confidence: confidence.value });
    if (recordedAt.value) params.set('recorded_at', new Date(recordedAt.value).toISOString());
    request.open('POST', `/api/predict?${params}`);
    request.upload.addEventListener('progress', event => {
      if (event.lengthComputable) {
        const percent = Math.min(100, Math.round(event.loaded / event.total * 100));
        updateProgress('Subiendo archivo', 'Transfiriendo el archivo al procesador local.',
          percent, `${(event.loaded / 1048576).toFixed(1)} de ${(event.total / 1048576).toFixed(1)} MB`);
      }
    });
    request.addEventListener('load', () => {
      let data;
      try { data = JSON.parse(request.responseText); }
      catch { return reject(new Error('Respuesta inesperada del servidor.')); }
      if (request.status < 200 || request.status >= 300) {
        return reject(new Error(data.detail || 'No se pudo cargar el archivo.'));
      }
      resolve(data);
    });
    request.addEventListener('error', () => reject(new Error('No se pudo conectar con RoadGuard.')));
    const body = new FormData(); body.append('file', file);
    if (trackInput.files[0]) body.append('track', trackInput.files[0]);
    request.send(body);
  });
}
async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || 'No se pudo consultar el análisis.');
    if (job.status === 'completed') {
      sessionStorage.removeItem('roadguard-job');
      button.disabled = false;
      renderResults(job.result);
      setMessage('Análisis completado. Revisa las detecciones antes de tomar decisiones.');
      return;
    }
    if (job.status === 'failed') throw new Error(job.error || 'El análisis falló.');
    const total = job.total_frames;
    const current = job.processed_frames || 0;
    const percent = total ? Math.min(99, Math.round(current / total * 100)) : null;
    updateProgress(job.status === 'queued' ? 'Esperando turno' : 'Detectando daños',
      job.status === 'queued' ? 'Otro archivo se está procesando.' : 'Analizando fotogramas con el modelo B.',
      percent, total ? `${current} de ${total} fotogramas · ${job.total_detections || 0} detecciones` :
        `${current} fotogramas · ${job.total_detections || 0} detecciones`);
    timer = setTimeout(() => pollJob(jobId), 1200);
  } catch (error) {
    sessionStorage.removeItem('roadguard-job');
    button.disabled = false;
    setMessage(error.message, true);
    resultsCard.querySelector('.progress-track')?.classList.remove('indeterminate');
    document.querySelector('#progress-title').textContent = 'No se pudo completar';
    document.querySelector('#progress-note').textContent = error.message;
  }
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) return setMessage('Selecciona una imagen o video.', true);
  if (file.size > 500 * 1048576) return setMessage('El archivo supera el límite de 500 MB.', true);
  if (timer) clearTimeout(timer);
  button.disabled = true; details.classList.add('hidden'); setMessage('');
  processingView('Preparando archivo', 'Iniciando carga local.');
  try {
    const job = await upload(file);
    sessionStorage.setItem('roadguard-job', job.job_id);
    pollJob(job.job_id);
  } catch (error) {
    button.disabled = false;
    setMessage(error.message, true);
    document.querySelector('#progress-title').textContent = 'No se pudo cargar';
    document.querySelector('#progress-note').textContent = error.message;
  }
});
function renderResults(data) {
  resultsCard.className = 'results-card';
  const count = Number(data.total_detections) || 0;
  resultsCard.innerHTML = `<div class="section-head"><div><p class="eyebrow">ANÁLISIS COMPLETADO</p><h2>${count} detección${count === 1 ? '' : 'es'} encontrada${count === 1 ? '' : 's'}</h2></div><span class="step">${Number(data.confidence_threshold).toFixed(2)} CONF</span></div><p class="lede">${data.media_type === 'video' ? 'Las detecciones se cuentan por fotograma; un mismo daño puede aparecer varias veces.' : 'Revisa la evidencia y el resumen por tipo de daño.'}</p>`;
  details.classList.remove('hidden');
  reportLink.href = data.report_url;
  const summary = data.detections_by_class || {};
  summaryGrid.innerHTML = `<div class="summary-item total"><strong>${count}</strong><span>TOTAL</span></div>` +
    Object.entries(summary).map(([name, value]) => `<div class="summary-item"><strong>${Number(value)}</strong><span>${escapeHtml(CLASS_LABELS[name] || name)}<small> (${escapeHtml(name)})</small></span></div>`).join('');
  locationResult.replaceChildren();
  const title = document.createElement('strong');
  title.textContent = data.location ? 'Ubicación aproximada de la cámara' : 'Ubicación no disponible';
  locationResult.append(title);
  const note = document.createElement('p');
  if (data.location) {
    const { latitude, longitude, source } = data.location;
    const lat = Number(latitude), lon = Number(longitude);
    note.textContent = `${lat.toFixed(6)}, ${lon.toFixed(6)} · ${source}`;
    locationResult.append(note);
    if (data.media_type === 'image' && Number.isFinite(lat) && Number.isFinite(lon) &&
        Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
      const showMap = document.createElement('button');
      showMap.type = 'button';
      showMap.className = 'map-button';
      showMap.textContent = 'Mostrar mapa con pin';
      const privacy = document.createElement('small');
      privacy.textContent = 'Al mostrar el mapa, el navegador enviará estas coordenadas a OpenStreetMap.';
      locationResult.append(privacy, showMap);
      showMap.addEventListener('click', () => {
        const delta = 0.003;
        const bbox = [lon - delta, lat - delta, lon + delta, lat + delta].join(',');
        const map = document.createElement('iframe');
        map.className = 'location-map';
        map.title = 'Mapa con pin en la ubicación aproximada de la cámara';
        map.referrerPolicy = 'no-referrer';
        map.src = `https://www.openstreetmap.org/export/embed.html?bbox=${encodeURIComponent(bbox)}&layer=mapnik&marker=${encodeURIComponent(`${lat},${lon}`)}`;
        showMap.replaceWith(map);
      }, { once: true });
    }
    const link = document.createElement('a');
    link.href = `https://www.openstreetmap.org/?mlat=${encodeURIComponent(latitude)}&mlon=${encodeURIComponent(longitude)}#map=17/${encodeURIComponent(latitude)}/${encodeURIComponent(longitude)}`;
    link.target = '_blank'; link.rel = 'noopener noreferrer';
    link.textContent = 'Ver en mapa ↗';
    locationResult.append(link);
    const caution = document.createElement('small');
    caution.textContent = data.media_type === 'video' ? `Referencia de ${data.located_frames} fotogramas. El JSON contiene coordenadas por fotograma; son aproximadas y corresponden a la cámara.` : 'La coordenada es de la cámara, no necesariamente del daño.';
    locationResult.append(caution);
  } else {
    note.textContent = data.location_status === 'gpx_missing_time' ? 'El GPX requiere la fecha y hora de inicio de grabación para sincronizarlo.' : data.location_status === 'gpx_missing_fps' ? 'No se pudo leer la velocidad de fotogramas para sincronizar el GPX.' : 'El archivo no contiene GPS utilizable. Para videos puedes añadir un GPX.';
    locationResult.append(note);
  }
  mediaWrap.replaceChildren();
  details.querySelector('.media-download')?.remove();
  const media = document.createElement(data.media_type === 'video' ? 'video' : 'img');
  media.src = data.result_url;
  if (data.media_type === 'video') {
    media.controls = true;
    media.addEventListener('error', () => {
      mediaWrap.replaceChildren();
      const explanation = document.createElement('p');
      explanation.textContent = 'Tu navegador no reproduce este formato. Descarga el video para abrirlo en un reproductor local.';
      explanation.className = 'video-fallback';
      mediaWrap.append(explanation);
    }, { once: true });
  } else media.alt = 'Resultado de detección RoadGuard';
  mediaWrap.append(media);
  if (data.media_type === 'video') {
    const link = document.createElement('a');
    link.href = data.result_url;
    link.textContent = 'Descargar video procesado ↓';
    link.className = 'media-download';
    link.download = 'roadguard-resultado';
    mediaWrap.before(link);
  }
  details.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
const pending = sessionStorage.getItem('roadguard-job');
if (pending) {
  button.disabled = true;
  processingView('Recuperando análisis', 'Consultando el trabajo anterior.');
  pollJob(pending);
}
