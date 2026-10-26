"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";
const FRAME_INTERVAL_MS = 400;
type Prediction = { label: string; confidence: number; box: number[] };

export function SignCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const busyRef = useRef(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Kamera belum terhubung");
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  useEffect(() => {
    if (!running) return;
    let stream: MediaStream | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let active = true;

    const predict = async () => {
      if (!active) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2 || busyRef.current) {
        timer = setTimeout(predict, FRAME_INTERVAL_MS);
        return;
      }
      busyRef.current = true;
      try {
        canvas.width = Math.min(video.videoWidth, 640);
        canvas.height = Math.round(video.videoHeight * (canvas.width / video.videoWidth));
        canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.75));
        if (!blob) throw new Error("Gagal mengambil frame kamera");
        const form = new FormData();
        form.append("image", blob, "frame.jpg");
        const response = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail ?? "Prediksi gagal");
        if (active) {
          const nextPredictions = body.predictions as Prediction[];
          setPredictions(nextPredictions);
          setError("");
          setStatus(nextPredictions.length ? "Gestur terdeteksi" : "Tangan belum terdeteksi");
        }
      } catch (cause) {
        if (active) {
          setError(cause instanceof Error ? cause.message : "Tidak dapat menghubungi backend");
          setStatus("Backend tidak tersedia");
        }
      } finally {
        busyRef.current = false;
        if (active) timer = setTimeout(predict, FRAME_INTERVAL_MS);
      }
    };

    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Kamera perlu localhost atau HTTPS");
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
        if (!active) { stream.getTracks().forEach((track) => track.stop()); return; }
        if (!videoRef.current) throw new Error("Pratinjau kamera tidak tersedia");
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setStatus("Kamera aktif — mencari gestur");
        predict();
      } catch (cause) {
        if (active) {
          setError(cause instanceof Error ? cause.message : "Akses kamera gagal");
          setStatus("Kamera gagal dibuka");
          setRunning(false);
        }
      }
    };
    start();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
      stream?.getTracks().forEach((track) => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      busyRef.current = false;
    };
  }, [running]);

  const best = predictions.reduce<Prediction | null>((current, item) => !current || item.confidence > current.confidence ? item : current, null);

  return (
    <div className="sign-camera">
      <div className="camera-stage-topline">
        <span><i className={`status-dot ${running ? "" : "is-pending"}`} /> {status}</span>
        <span className="detection-badge">{best ? `${best.label} · ${Math.round(best.confidence * 100)}%` : "MENUNGGU GESTUR"}</span>
      </div>
      <video ref={videoRef} className={`sign-camera-video${running ? " is-active" : ""}`} muted playsInline aria-label="Pratinjau kamera SIBI" />
      {!running && (
        <div className="sign-camera-placeholder" aria-hidden="true">
          <span className="gesture-frame-label">AREA DETEKSI</span>
          <span className="gesture-frame-mark">+</span>
          <span className="sign-camera-prompt">Aktifkan kamera untuk mulai deteksi</span>
        </div>
      )}
      <canvas ref={canvasRef} hidden />
      {error && <p className="sign-camera-error" role="alert">{error}</p>}
      <div className="camera-stage-bottomline">
        <span aria-live="polite">{best ? `Huruf terdeteksi: ${best.label}` : "Deteksi pose alfabet statis A–Z"}</span>
        <button className="stage-control" type="button" onClick={() => { setError(""); setPredictions([]); setRunning((value) => !value); }}>
          {running ? "Matikan kamera" : "Mulai kamera"}
        </button>
      </div>
    </div>
  );
}
