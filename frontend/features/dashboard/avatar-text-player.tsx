"use client";

import { useRef, useState } from "react";
import SignAvatar, { type SignAvatarHandle } from "@/components/sign-avatar";

export function AvatarTextPlayer() {
  const api = useRef<SignAvatarHandle>(null);
  const [text, setText] = useState("A");
  const [status, setStatus] = useState("");
  const play = async () => {
    const value = text;
    if (!/^[a-zA-Z ]{1,60}$/.test(value)) {
      setStatus("Masukkan huruf A-Z dan spasi saja.");
      return;
    }
    setStatus("Memutar pose eksperimen...");
    const completed = await api.current?.playSequence([...value.toUpperCase().replace(/\s/g, "")], { fingerspellFallback: false });
    setStatus(completed ? "Sequence pose eksperimen selesai." : "Pose belum tersedia atau pemutaran dibatalkan.");
  };
  return (
    <div className="avatar-text-player">
      <div className="avatar-model-stage">
        <SignAvatar apiRef={api} autoPlay={false} height={320} background="#104b46" startBone="RightHand" posesUrl="/poses/prototype-a.json" onWarning={setStatus} onFinish={() => setStatus("Pose selesai diputar.")} />
      </div>
      <label htmlFor="avatar-spelling-text">Teks alfabet untuk avatar</label>
      <input id="avatar-spelling-text" className="text-field" value={text} maxLength={60} onChange={(event) => setText(event.target.value)} />
      <div className="avatar-playback-actions">
        <button className="primary-button" type="button" onClick={() => void play()}>Eja teks</button>
        <button className="stage-control" type="button" onClick={() => { api.current?.stop(); setStatus("Pemutaran dihentikan."); }}>Berhenti</button>
      </div>
      <p className="avatar-data-notice">Avatar bergerak hanya saat Anda memasukkan teks di panel ini dan menekan Eja teks. Pose A masih eksperimental; kamera tidak mengendalikan avatar.</p>
      {status && <p className="avatar-playback-status" role="status">{status}</p>}
    </div>
  );
}
