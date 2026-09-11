const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type HealthResponse = {
  status: string;
  service: string;
  version: string;
};

export async function getBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/v1/health`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Backend responded with ${response.status}`);
  }

  return response.json() as Promise<HealthResponse>;
}