"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";
const INTERVAL_MS = 300;

type Prediction = { label: string; confidence: number; box: number[] };

export function SignCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const busyRef = useRef(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  useEffect(() => {
    if (!running) return;
    let stream: MediaStream;
    let timer: ReturnType<typeof setTimeout>;
    let active = true;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (!active) { stream.getTracks().forEach((track) => track.stop()); return; }
        if (videoRef.current) videoRef.current.srcObject = stream;
        const predict = async () => {
          if (!active) return;
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (!video || !canvas || video.readyState < 2 || busyRef.current) {
            timer = setTimeout(predict, INTERVAL_MS);
            return;
          }
          busyRef.current = true;
          try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext("2d")?.drawImage(video, 0, 0);
            const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
            if (!blob) throw new Error("Gagal mengambil gambar kamera");
            const form = new FormData();
            form.append("image", blob, "frame.jpg");
            const response = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
            if (!response.ok) throw new Error((await response.json()).detail ?? "Prediksi gagal");
            const result: { predictions: Prediction[] } = await response.json();
            if (active) { setPredictions(result.predictions); setError(""); }
          } catch (cause) {
            if (active) setError(cause instanceof Error ? cause.message : "Koneksi backend gagal");
          } finally {
            busyRef.current = false;
            if (active) timer = setTimeout(predict, INTERVAL_MS);
          }
        };
        await videoRef.current?.play();
        predict();
      } catch (cause) {
        if (active) { setError(cause instanceof Error ? cause.message : "Kamera tidak tersedia"); setRunning(false); }
      }
    }

    start();
    return () => {
      active = false;
      clearTimeout(timer);
      stream?.getTracks().forEach((track) => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
    };
  }, [running]);

  const best = predictions.reduce<Prediction | null>((top, item) => !top || item.confidence > top.confidence ? item : top, null);

  return (
    <div>
      <div className="camera-preview" style={{ position: "relative" }}>
        <video ref={videoRef} muted playsInline aria-label="Pratinjau kamera untuk deteksi SIBI" style={{ display: running ? "block" : "none", width: "100%" }} />
        <canvas ref={canvasRef} hidden />
        {!running && <p>Izinkan akses kamera untuk mulai deteksi.</p>}
        {best && <output aria-live="polite">{best.label} ({Math.round(best.confidence * 100)}%)</output>}
      </div>
      <p role="status">{error || (running ? best ? `Gestur terdeteksi: ${best.label}` : "Mencari gestur…" : "Kamera belum terhubung")}</p>
      <button className="stage-control" type="button" onClick={() => { setError(""); setRunning((value) => !value); }}>
        {running ? "Matikan kamera" : "Mulai kamera"}
      </button>
    </div>
  );
}
