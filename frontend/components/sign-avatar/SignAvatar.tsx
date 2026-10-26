"use client";

import {
  Component,
  Suspense,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { AvatarModel } from "./AvatarModel";
import { DebugPanel } from "./DebugPanel";
import { parsePoseLibrary } from "./poses";
import type { Rig } from "./rig";
import type { SequencePlayer } from "./player";
import type { PlayOptions, PoseEntry, PoseLibrary, SignAvatarProps } from "./types";

// Kamera fokus tubuh bagian atas (model dinormalisasi 1.7 m, kaki di y=0).
// Lebar pandang ≈ 1.1 m di jarak 2 m: kepala, dada, dan area isyarat di depan badan terlihat.
const CAMERA_POSITION: [number, number, number] = [0, 1.3, 2.0];
const CAMERA_TARGET: [number, number, number] = [0, 1.2, 0];

/** string → dieja per huruf; array → dipakai apa adanya */
const toTokens = (gloss?: string | string[]): string[] =>
  Array.isArray(gloss)
    ? gloss.map((g) => g.trim()).filter(Boolean)
    : [...(gloss ?? "").replace(/\s+/g, "")];

/** Tangkap error pemuatan model (mis. avatar.glb tidak ada) agar halaman tidak crash. */
class ModelErrorBoundary extends Component<{ url: string; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" style={{ padding: 16, color: "#ffb4b4", font: "14px system-ui" }}>
        Gagal memuat model <code>{this.props.url}</code>. Pastikan file .glb ada di folder public.
      </div>
    );
  }
}

export default function SignAvatar({
  gloss,
  speed = 1,
  gapMs = 150,
  autoPlay = true,
  paused = false,
  onFinish,
  onWarning,
  modelUrl = "/models/avatar.glb",
  height = 360,
  startBone = "RightHand",
  bodyPoseRef,
  poses: posesProp,
  posesUrl = "/poses/sibi-poses.json",
  boneNameMap,
  restKey = "_REST",
  fingerspellFallback = false,
  debug = false,
  background = "#2b323c",
  className,
  apiRef,
}: SignAvatarProps) {
  const [fetched, setFetched] = useState<PoseLibrary>({});
  const [saved, setSaved] = useState<PoseLibrary>({}); // pose yang disimpan dari panel debug
  const [fetchedReady, setFetchedReady] = useState(false);
  const posesReady = !!posesProp || fetchedReady;
  const [warnings, setWarnings] = useState<string[]>([]);
  const [rig, setRig] = useState<Rig | null>(null);
  const playerRef = useRef<SequencePlayer | null>(null);
  const previousPaused = useRef(paused);

  // nilai terbaru disimpan di ref agar callback stabil (tidak memicu efek berulang)
  const latest = useRef({ speed, gapMs, fingerspell: fingerspellFallback, debug, onFinish, onWarning });
  useEffect(() => {
    latest.current = { speed, gapMs, fingerspell: fingerspellFallback, debug, onFinish, onWarning };
  }, [speed, gapMs, fingerspellFallback, debug, onFinish, onWarning]);

  const warn = useCallback((msg: string | string[]) => {
    const list = (Array.isArray(msg) ? msg : [msg]).filter(Boolean);
    if (!list.length) return;
    setWarnings((prev) => Array.from(new Set([...prev, ...list])).slice(-6));
    list.forEach((m) => {
      console.warn("[SignAvatar]", m);
      latest.current.onWarning?.(m);
    });
  }, []);

  // --- muat data pose (prop `poses` atau fetch dari posesUrl) ---
  useEffect(() => {
    if (posesProp) return;
    let cancelled = false;
    fetch(posesUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json) => {
        if (cancelled) return;
        const { poses, warnings: w } = parsePoseLibrary(json);
        setFetched(poses);
        setFetchedReady(true);
        warn(w);
      })
      .catch((e: Error) => !cancelled && warn(`Gagal memuat ${posesUrl}: ${e.message}`))
      .finally(() => !cancelled && setFetchedReady(true));
    return () => {
      cancelled = true;
    };
  }, [posesProp, posesUrl, warn]);

  const library = useMemo<PoseLibrary>(() => ({ ...fetched, ...saved }), [fetched, saved]);

  const onReady = useCallback((r: Rig | null, p: SequencePlayer | null) => {
    playerRef.current = p;
    setRig(r);
  }, []);

  // sinkronkan library ke player (dideklarasikan sebelum efek autoPlay)
  useEffect(() => {
    playerRef.current?.configure(library, restKey);
  }, [library, restKey, rig]);

  // --- API pemutaran ---
  const runSequence = useCallback(
    async (tokens: string[], opts: PlayOptions = {}) => {
      const player = playerRef.current;
      const cur = latest.current;
      if (!player) {
        warn("Avatar belum selesai dimuat.");
        return false;
      }
      if (cur.debug) {
        warn("Mode debug aktif: pemutaran dimatikan.");
        return false;
      }
      setWarnings([]);
      const completed = await player.play(tokens, {
        speed: opts.speed ?? cur.speed,
        gapMs: opts.gapMs ?? cur.gapMs,
        returnToRest: opts.returnToRest ?? true,
        fingerspell: opts.fingerspellFallback ?? cur.fingerspell,
      });
      if (completed) latest.current.onFinish?.();
      return completed;
    },
    [warn]
  );

  useImperativeHandle(
    apiRef,
    () => ({
      playSequence: runSequence,
      stop: () => playerRef.current?.stop(true),
      isPlaying: () => !!playerRef.current?.playing,
    }),
    [runSequence]
  );

  // putar otomatis saat gloss berubah
  const glossKey = Array.isArray(gloss) ? gloss.join("\u0001") : gloss ?? "";
  useEffect(() => {
    if (!autoPlay || !rig || !posesReady || debug) return;
    if (!glossKey) return playerRef.current?.stop(true);
    void runSequence(toTokens(gloss));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- glossKey mewakili isi `gloss`
  }, [glossKey, autoPlay, rig, posesReady, debug, runSequence]);

  // masuk mode debug: hentikan pemutaran yang sedang berjalan
  useEffect(() => {
    if (paused && !previousPaused.current) playerRef.current?.stop(true);
    previousPaused.current = paused;
  }, [paused]);

  useEffect(() => {
    if (debug) playerRef.current?.stop();
  }, [debug]);

  const savePose = useCallback((name: string, entry: PoseEntry) => {
    setSaved((p) => ({ ...p, [name]: entry }));
  }, []);

  return (
    <div
      className={className}
      style={{ position: "relative", width: "100%", height, minHeight: 240, background, overflow: "hidden" }}
    >
      <ModelErrorBoundary url={modelUrl} key={modelUrl}>
        <Canvas fallback={<div role="alert" className="avatar-load-error">WebGL tidak tersedia. Gunakan browser dengan akselerasi grafis aktif.</div>} camera={{ position: CAMERA_POSITION, fov: 30, near: 0.1, far: 20 }} dpr={[1, 2]}>
          {/* tanpa Environment (HDR dari CDN) agar jalan offline */}
          <hemisphereLight args={["#ffffff", "#3a4452", 0.9]} />
          <directionalLight position={[1.5, 2.5, 2.5]} intensity={2.2} />
          <directionalLight position={[-2, 1.5, 1]} intensity={0.8} color="#cfe0ff" />
          <directionalLight position={[0, 2, -2]} intensity={0.9} />

          <Suspense fallback={null}>
            <AvatarModel url={modelUrl} nameMap={boneNameMap} paused={debug || paused} onWarn={warn} onReady={onReady} startBone={startBone} bodyPoseRef={bodyPoseRef} />
          </Suspense>

          <OrbitControls target={CAMERA_TARGET} enablePan={false} enableDamping minDistance={0.8} maxDistance={3} />
        </Canvas>
      </ModelErrorBoundary>

      {warnings.length > 0 && (
        <div
          role="status"
          style={{
            position: "absolute",
            left: 8,
            bottom: 8,
            maxWidth: "min(420px, 70%)",
            padding: "8px 10px",
            background: "rgba(60,44,10,.92)",
            color: "#ffe6a8",
            font: "12px/1.4 system-ui",
            borderRadius: 6,
          }}
        >
          {warnings.map((w) => (
            <div key={w}>⚠ {w}</div>
          ))}
          <button
            onClick={() => setWarnings([])}
            style={{ marginTop: 6, background: "none", border: "1px solid #ffe6a8", color: "inherit", borderRadius: 4, cursor: "pointer" }}
          >
            Tutup
          </button>
        </div>
      )}

      {debug && rig && <DebugPanel rig={rig} library={library} restKey={restKey} onSavePose={savePose} />}
    </div>
  );
}
