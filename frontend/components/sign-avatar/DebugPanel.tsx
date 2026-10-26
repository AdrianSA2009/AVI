"use client";

import { useMemo, useState } from "react";
import * as THREE from "three";
import { DEBUG_GROUPS, type Rig } from "./rig";
import type { PoseEntry, PoseLibrary, Quat } from "./types";

/**
 * Slider = rotasi Euler (derajat, urutan XYZ) RELATIF terhadap bind pose model, pada sumbu
 * lokal tulang. 0,0,0 = pose bawaan. Saat diekspor, nilainya dikonversi ke quaternion
 * absolut (rotasi lokal tulang) sesuai format JSON.
 */
type Vec3 = [number, number, number];
const AXES = ["X", "Y", "Z"] as const;
const ZERO: Vec3 = [0, 0, 0];

const tmpE = new THREE.Euler();
const tmpQ = new THREE.Quaternion();
const tmpInv = new THREE.Quaternion();
const rad = THREE.MathUtils.degToRad;
const deg = THREE.MathUtils.radToDeg;
const round = (n: number, p: number) => Math.round(n * 10 ** p) / 10 ** p;

interface Props {
  rig: Rig;
  library: PoseLibrary;
  restKey: string;
  onSavePose: (name: string, entry: PoseEntry) => void;
}

const CSS = `
.sa-debug{position:absolute;top:0;right:0;bottom:0;width:min(340px,100%);overflow:auto;box-sizing:border-box;
  padding:12px;background:rgba(18,22,28,.92);color:#e8ecf1;font:12px/1.4 system-ui,sans-serif}
.sa-debug h3{margin:0 0 8px;font-size:13px}
.sa-debug label{display:block;margin:6px 0 2px;color:#aab4c0}
.sa-debug input[type=text],.sa-debug input[type=number],.sa-debug select{width:100%;box-sizing:border-box;padding:4px 6px;
  background:#0f1318;color:inherit;border:1px solid #3a4452;border-radius:4px}
.sa-debug button{margin:6px 6px 0 0;padding:4px 8px;background:#2f6fed;color:#fff;border:0;border-radius:4px;cursor:pointer}
.sa-debug button.sa-ghost{background:#2b3440}
.sa-debug details{margin-top:8px;border-top:1px solid #2b3440;padding-top:6px}
.sa-debug summary{cursor:pointer;font-weight:600}
.sa-bone{margin:8px 0}
.sa-bone-head{display:flex;justify-content:space-between;align-items:center}
.sa-bone-head button{margin:0;padding:1px 6px;font-size:11px}
.sa-row{display:grid;grid-template-columns:14px 1fr 44px;gap:6px;align-items:center}
.sa-row input{width:100%}
.sa-debug pre{max-height:160px;overflow:auto;background:#0f1318;padding:6px;border-radius:4px;font-size:11px}
.sa-status{margin-top:6px;color:#8fd19e}
`;

export function DebugPanel({ rig, library, restKey, onSavePose }: Props) {
  // hanya tampilkan tulang yang benar-benar ada di model
  const groups = useMemo(
    () =>
      DEBUG_GROUPS.map((g) => ({ ...g, bones: g.bones.filter((b) => rig.has(b)) })).filter(
        (g) => g.bones.length > 0
      ),
    [rig]
  );

  const [angles, setAngles] = useState<Record<string, Vec3>>({});
  const [poseName, setPoseName] = useState("A");
  const [duration, setDuration] = useState(0.6);
  const [status, setStatus] = useState("");

  /** Terapkan sudut (derajat) ke tulang: bind * delta */
  const apply = (name: string, v: Vec3) => {
    const bone = rig.get(name);
    const bind = rig.bindQuat(name);
    if (!bone || !bind) return;
    tmpE.set(rad(v[0]), rad(v[1]), rad(v[2]), "XYZ");
    bone.quaternion.copy(bind).multiply(tmpQ.setFromEuler(tmpE));
  };

  const setAxis = (name: string, axis: number, value: number) => {
    const next = [...(angles[name] ?? ZERO)] as Vec3;
    next[axis] = value;
    setAngles((p) => ({ ...p, [name]: next }));
    apply(name, next);
  };

  const resetBone = (name: string) => {
    setAngles((p) => ({ ...p, [name]: ZERO }));
    apply(name, ZERO);
  };

  const resetAll = () => {
    Object.keys(angles).forEach((n) => apply(n, ZERO));
    setAngles({});
  };

  /** Quaternion absolut semua tulang yang diubah (selain 0,0,0). */
  const collect = (): Record<string, Quat> => {
    const out: Record<string, Quat> = {};
    for (const [name, v] of Object.entries(angles)) {
      if (v.every((x) => x === 0)) continue;
      const bone = rig.get(name);
      if (!bone) continue;
      let { x, y, z, w } = bone.quaternion;
      if (w < 0) [x, y, z, w] = [-x, -y, -z, -w]; // tanda kanonik (rotasi sama)
      out[name] = [round(x, 5), round(y, 5), round(z, 5), round(w, 5)];
    }
    return out;
  };

  /** Muat pose dari library ke slider (quaternion absolut → delta dari bind). */
  const loadPose = (key: string) => {
    resetAll();
    const entry = library[key];
    if (!entry) return;
    const next: Record<string, Vec3> = {};
    for (const [name, q] of Object.entries(entry.bones)) {
      const bind = rig.bindQuat(name);
      if (!bind || !rig.has(name)) continue;
      // delta = bind^-1 * q  (kebalikan dari apply: q = bind * delta)
      tmpQ.set(q[0], q[1], q[2], q[3]).normalize();
      tmpQ.premultiply(tmpInv.copy(bind).invert());
      tmpE.setFromQuaternion(tmpQ, "XYZ");
      next[name] = [round(deg(tmpE.x), 1), round(deg(tmpE.y), 1), round(deg(tmpE.z), 1)];
    }
    Object.entries(next).forEach(([n, v]) => apply(n, v));
    setAngles(next);
    setPoseName(key);
    if (entry.duration) setDuration(entry.duration);
    setStatus(`Pose "${key}" dimuat.`);
  };

  const entryJson = JSON.stringify({ [poseName]: { bones: collect(), duration } }, null, 2);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(entryJson);
      setStatus("JSON pose disalin ke clipboard.");
    } catch {
      setStatus("Gagal menyalin; salin manual dari pratinjau.");
    }
  };

  const save = () => {
    const name = poseName.trim();
    if (!name) return setStatus("Isi nama pose dulu.");
    onSavePose(name, { bones: collect(), duration });
    setStatus(`Pose "${name}" disimpan ke library (sementara, belum ke file).`);
  };

  const download = () => {
    const blob = new Blob([JSON.stringify(library, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sibi-poses.json";
    a.click();
    URL.revokeObjectURL(url);
    setStatus("Seluruh library diunduh.");
  };

  return (
    <aside className="sa-debug" aria-label="Panel debug pose">
      <style>{CSS}</style>
      <h3>Mode debug pose</h3>

      <label htmlFor="sa-load">Muat pose dari library</label>
      <select id="sa-load" value="" onChange={(e) => e.target.value && loadPose(e.target.value)}>
        <option value="">— pilih —</option>
        {Object.keys(library).map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>

      <label htmlFor="sa-name">Nama pose (huruf / gloss; &quot;{restKey}&quot; untuk pose istirahat)</label>
      <input id="sa-name" type="text" value={poseName} onChange={(e) => setPoseName(e.target.value)} />

      <label htmlFor="sa-dur">Durasi transisi (detik)</label>
      <input
        id="sa-dur"
        type="number"
        min={0.1}
        step={0.1}
        value={duration}
        onChange={(e) => setDuration(Math.max(0.1, Number(e.target.value) || 0.6))}
      />

      <div>
        <button onClick={save}>Simpan ke library</button>
        <button onClick={copy}>Salin JSON pose</button>
        <button className="sa-ghost" onClick={download}>
          Unduh semua JSON
        </button>
        <button className="sa-ghost" onClick={resetAll}>
          Reset slider
        </button>
      </div>
      {status && <div className="sa-status" role="status">{status}</div>}

      <details>
        <summary>Pratinjau JSON pose</summary>
        <pre>{entryJson}</pre>
      </details>

      {groups.length === 0 && (
        <p>Tidak ada tulang yang cocok di model. Isi prop boneNameMap (nama JSON → nama node).</p>
      )}

      {groups.map((g, i) => (
        <details key={g.label} open={i === 0}>
          <summary>{g.label}</summary>
          {g.bones.map((name) => {
            const v = angles[name] ?? ZERO;
            return (
              <div className="sa-bone" key={name}>
                <div className="sa-bone-head">
                  <span>{name}</span>
                  <button className="sa-ghost" onClick={() => resetBone(name)}>
                    reset
                  </button>
                </div>
                {AXES.map((ax, axis) => (
                  <div className="sa-row" key={ax}>
                    <span>{ax}</span>
                    <input
                      type="range"
                      min={-180}
                      max={180}
                      step={1}
                      value={v[axis]}
                      aria-label={`${name} sumbu ${ax}`}
                      onChange={(e) => setAxis(name, axis, Number(e.target.value))}
                    />
                    <span>{v[axis]}°</span>
                  </div>
                ))}
              </div>
            );
          })}
        </details>
      ))}
    </aside>
  );
}
