// Centralized API configuration for local dev and cloud production deployments (Vercel)
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
