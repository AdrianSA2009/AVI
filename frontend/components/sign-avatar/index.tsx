"use client";

import dynamic from "next/dynamic";

// next/dynamic dengan ssr:false hanya boleh di Client Component (App Router)
const SignAvatar = dynamic(() => import("./SignAvatar"), {
  ssr: false,
  loading: () => (
    <div className="avatar-loading" role="status">Memuat avatar…</div>
  ),
});

export default SignAvatar;
export type {
  SignAvatarProps,
  SignAvatarHandle,
  PlayOptions,
  PoseEntry,
  PoseLibrary,
  Quat,
} from "./types";
