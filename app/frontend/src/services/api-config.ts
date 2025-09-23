const DEFAULT_API_PORT = 8000;
const DEFAULT_API_PROTOCOL = 'http:';
const DEFAULT_API_HOSTNAME = 'localhost';

const buildUrlFromWindow = () => {
  if (typeof window === 'undefined') {
    return null;
  }

  const protocol = window.location.protocol || DEFAULT_API_PROTOCOL;
  const hostname = window.location.hostname || DEFAULT_API_HOSTNAME;

  return `${protocol}//${hostname}:${DEFAULT_API_PORT}`;
};

export const getApiBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL?.trim();
  if (envUrl) {
    return envUrl;
  }

  const windowUrl = buildUrlFromWindow();
  if (windowUrl) {
    return windowUrl;
  }

  return `${DEFAULT_API_PROTOCOL}//${DEFAULT_API_HOSTNAME}:${DEFAULT_API_PORT}`;
};

export const API_BASE_URL = getApiBaseUrl();
