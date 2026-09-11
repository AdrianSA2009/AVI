"use client";

import { useEffect, useState } from "react";

import { getBackendHealth, type HealthResponse } from "@/lib/api";

type ConnectionState =
  | { status: "checking" }
  | { status: "connected"; data: HealthResponse }
  | { status: "offline"; message: string };

export function BackendStatus() {
  const [connection, setConnection] = useState<ConnectionState>({
    status: "checking",
  });

  useEffect(() => {
    let active = true;

    getBackendHealth()
      .then((data) => {
        if (active) setConnection({ status: "connected", data });
      })
      .catch((error: unknown) => {
        if (active) {
          setConnection({
            status: "offline",
            message: error instanceof Error ? error.message : "Request failed",
          });
        }
      });

    return () => {
      active = false;
    };
  }, []);

  if (connection.status === "checking") {
    return <p className="status statusChecking">Menghubungkan ke backend...</p>;
  }

  if (connection.status === "offline") {
    return (
      <div className="status statusOffline">
        <strong>Backend belum terhubung</strong>
        <span>{connection.message}</span>
      </div>
    );
  }

  return (
    <div className="status statusConnected">
      <strong>Backend terhubung</strong>
      <span>
        {connection.data.service} v{connection.data.version}
      </span>
    </div>
  );
}