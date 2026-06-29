import { useState, useEffect } from 'react';

// ─── Utility helpers ──────────────────────────────────────────────────────────

function safeWidth(lx, rx) {
  try {
    const w = Math.abs(parseFloat(rx) - parseFloat(lx));
    return isNaN(w) ? '—' : `${w.toFixed(1)} ft`;
  } catch {
    return '—';
  }
}

function formatArea(val) {
  if (val == null) return '—';
  return `${val.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} sq ft`;
}

function formatVol(val) {
  if (val == null || isNaN(val)) return '—';
  return val.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 1 });
}

/**
 * Parse station strings like "12+50", "STA 12+50", "1250" → feet as a number.
 */
function parseStation(stationStr) {
  if (!stationStr) return null;
  const clean = String(stationStr)
    .replace(/sta\.?/gi, '')
    .replace(/[^0-9+.]/g, '')
    .trim();
  if (clean.includes('+')) {
    const [left, right] = clean.split('+');
    const l = parseFloat(left);
    const r = parseFloat(right);
    if (isNaN(l) || isNaN(r)) return null;
    return l * 100 + r;
  }
  const val = parseFloat(clean);
  return isNaN(val) ? null : val;
}

/**
 * Classify a region as 'LEFT' or 'RIGHT'.
 * CROSSES_CENTERLINE → whichever catch-point x absolute value is larger.
 */
function getRegionSide(region) {
  const side = (region.side || '').toUpperCase();
  if (side === 'LEFT') return 'LEFT';
  if (side === 'RIGHT') return 'RIGHT';
  // CROSSES_CENTERLINE or unknown → use catch-point x magnitudes
  const lcpX = Math.abs(parseFloat(region.left_catch_point?.x) || 0);
  const rcpX = Math.abs(parseFloat(region.right_catch_point?.x) || 0);
  return lcpX >= rcpX ? 'LEFT' : 'RIGHT';
}

/**
 * Sum areas from intersections list by type + side.
 * Returns { lc, lf, rc, rf } (left-cut, left-fill, right-cut, right-fill).
 */
function computeAreas(intersections) {
  let lc = 0, lf = 0, rc = 0, rf = 0;
  (intersections || []).forEach((r) => {
    const area = r.area_calculation?.total_area_sqft || 0;
    const type = (r.type || '').toUpperCase();
    const side = getRegionSide(r);
    if (type === 'CUT') {
      if (side === 'LEFT') lc += area; else rc += area;
    } else if (type === 'FILL') {
      if (side === 'LEFT') lf += area; else rf += area;
    }
  });
  return { lc, lf, rc, rf };
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function VerticesTable({ vertices }) {
  const [open, setOpen] = useState(false);
  if (!vertices || vertices.length === 0) return null;

  return (
    <div className="expandable">
      <button className="expand-btn" onClick={() => setOpen(!open)}>
        <span className={`expand-arrow ${open ? 'open' : ''}`}>▶</span>
        📐 Polygon Vertices ({vertices.length} points)
      </button>
      {open && (
        <div className="expand-content">
          <table className="sample-table">
            <thead>
              <tr>
                <th>Offset (ft)</th>
                <th>Elevation (ft)</th>
              </tr>
            </thead>
            <tbody>
              {vertices.map((v, i) => (
                <tr key={i}>
                  <td>{v.x ?? '—'}</td>
                  <td>{v.elevation ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function IntersectionCard({ region, index }) {
  const rid = region.id ?? index + 1;
  const rtype = (region.type || 'UNKNOWN').toUpperCase();
  const lcp = region.left_catch_point || {};
  const rcp = region.right_catch_point || {};
  const areaCalc = region.area_calculation || {};
  const totalArea = areaCalc.total_area_sqft;
  const vertices  = areaCalc.vertices || [];
  const notes = region.notes;

  let cardClass, icon, desc;
  if (rtype === 'CUT') {
    cardClass = 'cut';
    icon = '🔴';
    desc = 'Proposed grade is BELOW existing ground — excavation needed';
  } else if (rtype === 'FILL') {
    cardClass = 'fill';
    icon = '🟢';
    desc = 'Proposed grade is ABOVE existing ground — fill material needed';
  } else {
    cardClass = 'unknown';
    icon = '⚪';
    desc = 'Undetermined';
  }

  return (
    <div className={`intersection-card ${cardClass}`}>
      <div className="intersection-id">Intersection {rid}</div>
      <div className={`intersection-type ${cardClass}`}>
        {icon} {rtype}
      </div>
      <div className="intersection-desc">{desc}</div>

      <div className="info-grid">
        <div className="info-item">
          <div className="k">Left Catch Point</div>
          <div className="v">
            x = {lcp.x ?? '—'} ft &nbsp;|&nbsp; elev = {lcp.elevation ?? '—'} ft
          </div>
        </div>
        <div className="info-item">
          <div className="k">Right Catch Point</div>
          <div className="v">
            x = {rcp.x ?? '—'} ft &nbsp;|&nbsp; elev = {rcp.elevation ?? '—'} ft
          </div>
        </div>
        <div className="info-item">
          <div className="k">Width</div>
          <div className="v">{safeWidth(lcp.x, rcp.x)}</div>
        </div>
        <div className="info-item">
          <div className="k">Area (Shoelace Formula)</div>
          <div className="v large">{formatArea(totalArea)}</div>
        </div>
      </div>

      <VerticesTable vertices={vertices} />

      {notes && (
        <div className="info-item" style={{ marginTop: '0.5rem' }}>
          <div className="k">Notes</div>
          <div className="v">{notes}</div>
        </div>
      )}
    </div>
  );
}

function SummaryPanel({ intersections }) {
  const cutTotal = intersections
    .filter((r) => (r.type || '').toUpperCase() === 'CUT')
    .reduce((s, r) => s + ((r.area_calculation || {}).total_area_sqft || 0), 0);

  const fillTotal = intersections
    .filter((r) => (r.type || '').toUpperCase() === 'FILL')
    .reduce((s, r) => s + ((r.area_calculation || {}).total_area_sqft || 0), 0);

  const net = cutTotal - fillTotal;
  const netClass = net > 0 ? 'net-cut' : net < 0 ? 'net-fill' : 'net-zero';

  return (
    <div className="summary-panel">
      <div className="summary-title">Analysis Summary</div>
      <div className="summary-row">
        <div className="summary-item">
          <div className="label">Total CUT</div>
          <div className="value cut">{formatArea(cutTotal)}</div>
        </div>
        <div className="summary-item">
          <div className="label">Total FILL</div>
          <div className="value fill">{formatArea(fillTotal)}</div>
        </div>
        <div className="summary-item">
          <div className="label">Net (CUT − FILL)</div>
          <div className={`value ${netClass}`}>
            {net >= 0 ? '+' : ''}{formatArea(Math.abs(net))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Volume Table ─────────────────────────────────────────────────────────────

function VolumeTable({ data }) {
  const intersections = data.intersections || [];

  // derive initial area values from Gemini result
  const init = computeAreas(intersections);

  const [currSta, setCurrSta]       = useState(data.station || '');
  const [prevSta, setPrevSta]       = useState(data.previous_station || '');
  const [leftCutArea,  setLeftCutArea]  = useState(init.lc.toFixed(1));
  const [leftFillArea, setLeftFillArea] = useState(init.lf.toFixed(1));
  const [rightCutArea,  setRightCutArea]  = useState(init.rc.toFixed(1));
  const [rightFillArea, setRightFillArea] = useState(init.rf.toFixed(1));

  // re-initialise whenever the page data changes
  useEffect(() => {
    const fresh = computeAreas(data.intersections || []);
    setCurrSta(data.station || '');
    setPrevSta(data.previous_station || '');
    setLeftCutArea(fresh.lc.toFixed(1));
    setLeftFillArea(fresh.lf.toFixed(1));
    setRightCutArea(fresh.rc.toFixed(1));
    setRightFillArea(fresh.rf.toFixed(1));
  }, [data]);

  // derived numbers
  const parsedCurr = parseStation(currSta);
  const parsedPrev = parseStation(prevSta);

  const distance =
    parsedCurr !== null && parsedPrev !== null
      ? Math.abs(parsedCurr - parsedPrev)
      : null;

  const lc = parseFloat(leftCutArea)  || 0;
  const lf = parseFloat(leftFillArea) || 0;
  const rc = parseFloat(rightCutArea)  || 0;
  const rf = parseFloat(rightFillArea) || 0;

  const lcVol = distance !== null ? lc * distance : null;
  const lfVol = distance !== null ? lf * distance : null;
  const rcVol = distance !== null ? rc * distance : null;
  const rfVol = distance !== null ? rf * distance : null;

  const totalCutVol  = lcVol  !== null && rcVol  !== null ? lcVol  + rcVol  : null;
  const totalFillVol = lfVol  !== null && rfVol  !== null ? lfVol  + rfVol  : null;

  const distLabel = distance !== null ? `${distance.toFixed(1)} ft` : '—';

  return (
    <div className="volume-card">
      <div className="volume-card-title">📐 Segment Volume Calculation (cu ft)</div>

      <div className="volume-table-wrapper">
        <table className="volume-profile-table">
          <thead>
            <tr>
              {/* Station group */}
              <th rowSpan="2" className="vt-center">Prev&nbsp;Station</th>
              <th rowSpan="2" className="vt-center">Current&nbsp;Station</th>
              <th rowSpan="2" className="vt-center vt-sep">Distance&nbsp;(ft)</th>

              {/* Area groups */}
              <th colSpan="2" className="vt-center vt-sep vt-group-left">Left Areas (sq ft)</th>
              <th colSpan="2" className="vt-center vt-sep vt-group-right">Right Areas (sq ft)</th>

              {/* Volume groups */}
              <th colSpan="2" className="vt-center vt-sep vt-group-left">Left Volumes (cu ft)</th>
              <th colSpan="2" className="vt-center vt-sep vt-group-right">Right Volumes (cu ft)</th>

              {/* Totals */}
              <th colSpan="2" className="vt-center vt-sep vt-group-total">Total Volumes (cu ft)</th>
            </tr>
            <tr>
              <th className="vt-sep">Cut</th>
              <th>Fill</th>
              <th className="vt-sep">Cut</th>
              <th>Fill</th>
              <th className="vt-sep vt-vol">Cut Vol</th>
              <th className="vt-vol">Fill Vol</th>
              <th className="vt-sep vt-vol">Cut Vol</th>
              <th className="vt-vol">Fill Vol</th>
              <th className="vt-sep vt-total">Total Cut</th>
              <th className="vt-total">Total Fill</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              {/* Prev station — read-only hint, editable for corrections */}
              <td>
                <input
                  type="text"
                  className="vt-input vt-input-sta"
                  value={prevSta}
                  onChange={(e) => setPrevSta(e.target.value)}
                  placeholder="e.g. 12+00"
                />
                <div className="vt-hint">
                  {parsedPrev !== null ? `${parsedPrev} ft` : 'invalid'}
                </div>
              </td>

              {/* Current station */}
              <td>
                <input
                  type="text"
                  className="vt-input vt-input-sta"
                  value={currSta}
                  onChange={(e) => setCurrSta(e.target.value)}
                  placeholder="e.g. 12+50"
                />
                <div className="vt-hint">
                  {parsedCurr !== null ? `${parsedCurr} ft` : 'invalid'}
                </div>
              </td>

              {/* Distance */}
              <td className="vt-sep vt-center vt-bold">{distLabel}</td>

              {/* Left areas */}
              <td className="vt-sep">
                <input
                  type="number"
                  className="vt-input vt-input-area"
                  value={leftCutArea}
                  min="0"
                  step="0.1"
                  onChange={(e) => setLeftCutArea(e.target.value)}
                />
              </td>
              <td>
                <input
                  type="number"
                  className="vt-input vt-input-area"
                  value={leftFillArea}
                  min="0"
                  step="0.1"
                  onChange={(e) => setLeftFillArea(e.target.value)}
                />
              </td>

              {/* Right areas */}
              <td className="vt-sep">
                <input
                  type="number"
                  className="vt-input vt-input-area"
                  value={rightCutArea}
                  min="0"
                  step="0.1"
                  onChange={(e) => setRightCutArea(e.target.value)}
                />
              </td>
              <td>
                <input
                  type="number"
                  className="vt-input vt-input-area"
                  value={rightFillArea}
                  min="0"
                  step="0.1"
                  onChange={(e) => setRightFillArea(e.target.value)}
                />
              </td>

              {/* Left volumes */}
              <td className="vt-sep vt-vol vt-right vt-bold vt-cut">{formatVol(lcVol)}</td>
              <td className="vt-vol vt-right vt-bold vt-fill">{formatVol(lfVol)}</td>

              {/* Right volumes */}
              <td className="vt-sep vt-vol vt-right vt-bold vt-cut">{formatVol(rcVol)}</td>
              <td className="vt-vol vt-right vt-bold vt-fill">{formatVol(rfVol)}</td>

              {/* Totals */}
              <td className="vt-sep vt-total vt-right vt-bold vt-cut">{formatVol(totalCutVol)}</td>
              <td className="vt-total vt-right vt-bold vt-fill">{formatVol(totalFillVol)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Summary row beneath table */}
      {distance !== null && (
        <div className="vt-summary-row">
          <div className="vt-summary-chip cut">
            <span className="vt-summary-label">Total Cut Volume</span>
            <span className="vt-summary-value">{formatVol(totalCutVol)} cu ft</span>
          </div>
          <div className="vt-summary-chip fill">
            <span className="vt-summary-label">Total Fill Volume</span>
            <span className="vt-summary-value">{formatVol(totalFillVol)} cu ft</span>
          </div>
          <div className="vt-summary-chip dist">
            <span className="vt-summary-label">Distance</span>
            <span className="vt-summary-value">{distLabel}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────

export default function AnalysisResult({ data }) {
  const [showRaw, setShowRaw] = useState(false);

  if (!data) return null;

  const station     = data.station || '—';
  const slopes      = data.slopes  || '—';
  const extent      = data.proposed_grade_extent || '—';
  const egLine      = data.existing_ground_line  || '—';
  const nRegions    = data.total_intersections   || 0;
  const intersections = data.intersections || [];

  // Show the volume table whenever previous_station is present
  // (it's attached by App.jsx after fetching the prev-page station)
  const hasPrevStation = data.hasOwnProperty('previous_station');

  return (
    <div className="result-panel">

      {/* Station header */}
      <div className="station-header">
        <div className="station-grid">
          <div className="station-item">
            <div className="label">Station</div>
            <div className="value highlight">{station}</div>
          </div>
          <div className="station-item">
            <div className="label">Slopes</div>
            <div className="value">{slopes}</div>
          </div>
          <div className="station-item">
            <div className="label">Proposed Grade Extent</div>
            <div className="value">{extent}</div>
          </div>
          <div className="station-item">
            <div className="label">Regions Found</div>
            <div className="value highlight">{nRegions}</div>
          </div>
        </div>
        <div className="station-divider">
          <span className="label">Existing Ground — </span>
          <span className="value">{egLine}</span>
        </div>
      </div>

      {/* Intersection cards */}
      {intersections.map((region, i) => (
        <IntersectionCard key={region.id ?? i} region={region} index={i} />
      ))}

      {/* Gemini area summary */}
      {intersections.length > 0 && <SummaryPanel intersections={intersections} />}

      {/* Volume Table — shown for pages > 1 (previous_station present) */}
      {hasPrevStation && <VolumeTable data={data} />}

      {/* Reasoning */}
      {data.reasoning && (
        <div className="section-block">
          <div className="section-hdr">Reasoning</div>
          <div className="section-content">{data.reasoning}</div>
        </div>
      )}

      {/* Notes */}
      {data.notes && (
        <div className="section-block">
          <div className="section-hdr">Notes</div>
          <div className="section-content">{data.notes}</div>
        </div>
      )}

      {/* Raw JSON (collapsible) */}
      <div className="expandable" style={{ marginTop: '1rem' }}>
        <button className="expand-btn" onClick={() => setShowRaw(!showRaw)}>
          <span className={`expand-arrow ${showRaw ? 'open' : ''}`}>▶</span>
          Raw JSON
        </button>
        {showRaw && (
          <div className="expand-content">
            <pre className="raw-json">{JSON.stringify(data, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
