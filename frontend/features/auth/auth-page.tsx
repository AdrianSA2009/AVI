"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";

type AuthPageProps = {
  mode: "login" | "register";
};

export function AuthPage({ mode }: AuthPageProps) {
  const isLogin = mode === "login";
  const [formMessage, setFormMessage] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!isLogin) {
      const formData = new FormData(event.currentTarget);
      if (formData.get("password") !== formData.get("confirmPassword")) {
        setFormMessage("Kata sandi belum cocok.");
        return;
      }
    }
    setFormMessage(
      isLogin
        ? "Login belum tersedia karena layanan autentikasi belum terhubung. Data Anda tidak dikirim."
        : "Pendaftaran belum tersedia karena layanan akun belum terhubung. Data Anda tidak dikirim.",
    );
  }

  return (
    <main className={`auth-layout${isLogin ? "" : " is-register"}`}>
      <aside className="auth-aside">
        <Link className="brand auth-brand" href="/login">
          <span className="brand-symbol" aria-hidden="true"><i /><i /></span>
          <span className="auth-brand-name">Translasi SIBI</span>
        </Link>
        <div className="auth-aside-content">
          <h2>{isLogin ? <>Isyarat jadi teks.<br />Suara jadi isyarat.</> : <>Daftar sekali.<br />Langsung bercakap.</>}</h2>
          <p className="auth-tagline">
            {isLogin
              ? "Percakapan dua arah antara pengguna SIBI dan pengguna Bahasa Indonesia."
              : "Pilih peran sesuai cara Anda berkomunikasi. Akun admin dibuat oleh pengelola sistem."}
          </p>
          <div className="auth-conversation" aria-label="Contoh percakapan dua arah">
            <div className="conversation-turn from-sign">
              <span>Dari isyarat</span>
              <p>Halo, nama saya Adrian.</p>
            </div>
            <div className="conversation-turn from-voice">
              <span>Dari suara</span>
              <p>Senang bertemu dengan Anda.</p>
            </div>
          </div>
        </div>
      </aside>

      <section className="auth-main" aria-labelledby="auth-title">
        <div className="auth-form-wrap">
          <h1 id="auth-title">{isLogin ? "Masuk ke akun Anda" : "Buat akun Anda"}</h1>
          <p className="auth-description">
            {isLogin
              ? "Admin, pengguna isyarat, dan pengguna umum masuk dari sini."
              : "Isi data diri, lalu pilih peran Anda."}
          </p>

          <form className="auth-form" onSubmit={handleSubmit}>
            {!isLogin && (
              <>
                <div className="auth-field">
                  <label className="field-label" htmlFor="full-name">Nama lengkap</label>
                  <input className="text-field" id="full-name" name="name" autoComplete="name" placeholder="Nama Anda" required />
                </div>
              </>
            )}
            <div className="auth-field">
              <label className="field-label" htmlFor="email">Email</label>
              <input className="text-field" id="email" name="email" type="email" autoComplete="email" placeholder="nama@email.com" required />
            </div>
            <div className="auth-field">
              <label className="field-label" htmlFor="password">Kata sandi</label>
              <input className="text-field" id="password" name="password" type="password" autoComplete={isLogin ? "current-password" : "new-password"} placeholder="••••••••••" minLength={8} required />
              <span className="password-hint">Minimal 8 karakter.</span>
            </div>
            {!isLogin && (
              <>
                <div className="auth-field">
                  <label className="field-label" htmlFor="confirm-password">Ulangi kata sandi</label>
                  <input className="text-field" id="confirm-password" name="confirmPassword" type="password" autoComplete="new-password" placeholder="••••••••" minLength={8} required />
                </div>
                <fieldset className="auth-role-fieldset">
                  <legend className="field-label">Saya mendaftar sebagai</legend>
                  <div className="auth-role-options">
                    <label className="auth-role-card">
                      <input type="radio" name="role" value="sign" defaultChecked />
                      <span className="auth-role-copy">
                        <strong>Pengguna bahasa isyarat</strong>
                        <small>Isyarat SIBI jadi teks dan suara</small>
                      </span>
                    </label>
                    <label className="auth-role-card">
                      <input type="radio" name="role" value="general" />
                      <span className="auth-role-copy">
                        <strong>Pengguna umum</strong>
                        <small>Suara atau teks jadi isyarat</small>
                      </span>
                    </label>
                  </div>
                </fieldset>
              </>
            )}

            <button className="primary-button auth-submit" type="submit" aria-describedby="auth-unavailable">
              {isLogin ? "Masuk" : "Buat akun"}
            </button>
          </form>

          <p className="auth-switch">
            {isLogin ? "Belum punya akun?" : "Sudah punya akun?"}{" "}
            <Link href={isLogin ? "/register" : "/login"}>{isLogin ? "Daftar" : "Masuk"}</Link>
          </p>
          <p className="auth-unavailable" id="auth-unavailable">Autentikasi belum terhubung. Form belum mengirim data.</p>
          {formMessage && <p className="auth-feedback" role="status" aria-live="polite">{formMessage}</p>}
        </div>
      </section>
    </main>
  );
}