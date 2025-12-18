/**
 * API client for the Diplomacy web app.
 */

import type {
  GameState,
  NewGameRequest,
  AdapterInfo,
  GameListItem,
  TrainingStats,
} from "./types";
import { ApiError } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// =============================================================================
// Error Types
// =============================================================================

export class NetworkError extends Error {
  readonly _tag = "NetworkError";
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

// =============================================================================
// Base Fetch Helpers
// =============================================================================

/**
 * Make a fetch request with error handling.
 */
async function fetchWithError(
  url: string,
  options?: RequestInit
): Promise<Response> {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        response.status,
        data.detail || `HTTP ${response.status}`
      );
    }
    return response;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new NetworkError(
      error instanceof Error ? error.message : "Network error"
    );
  }
}

// =============================================================================
// Game API
// =============================================================================

export const gameApi = {
  /**
   * Create a new game.
   */
  async createGame(request: NewGameRequest): Promise<GameState> {
    const response = await fetchWithError(`${API_BASE}/api/game/new`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    return response.json() as Promise<GameState>;
  },

  /**
   * Get current game state.
   */
  async getGame(gameId: string): Promise<GameState> {
    const response = await fetchWithError(`${API_BASE}/api/game/${gameId}`);
    return response.json() as Promise<GameState>;
  },

  /**
   * Get valid moves for human player.
   */
  async getValidMoves(gameId: string): Promise<Record<string, string[]>> {
    const response = await fetchWithError(
      `${API_BASE}/api/game/${gameId}/valid-moves`
    );
    return response.json();
  },

  /**
   * Submit orders and process AI moves.
   */
  async submitOrders(gameId: string, orders: string[]): Promise<GameState> {
    const response = await fetchWithError(
      `${API_BASE}/api/game/${gameId}/orders`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ orders }),
      }
    );
    return response.json() as Promise<GameState>;
  },

  /**
   * Delete a game.
   */
  async deleteGame(gameId: string): Promise<void> {
    await fetchWithError(`${API_BASE}/api/game/${gameId}`, {
      method: "DELETE",
    });
  },

  /**
   * List all games.
   */
  async listGames(): Promise<GameListItem[]> {
    const response = await fetchWithError(`${API_BASE}/api/game/list/all`);
    return response.json() as Promise<GameListItem[]>;
  },

  /**
   * Get available powers.
   */
  async getPowers(): Promise<string[]> {
    const response = await fetchWithError(`${API_BASE}/api/game/config/powers`);
    return response.json();
  },

  /**
   * Get available adapters (bot difficulties).
   */
  async getAdapters(): Promise<AdapterInfo[]> {
    const response = await fetchWithError(
      `${API_BASE}/api/game/config/adapters`
    );
    return response.json() as Promise<AdapterInfo[]>;
  },
};

// =============================================================================
// Training Data API
// =============================================================================

export const trainingApi = {
  /**
   * Get training data statistics.
   */
  async getStats(): Promise<TrainingStats> {
    const response = await fetchWithError(`${API_BASE}/api/training/stats`);
    return response.json() as Promise<TrainingStats>;
  },

  /**
   * Export training data.
   */
  async exportData(batchId?: string): Promise<{ count: number; trajectories: unknown[] }> {
    const url = batchId
      ? `${API_BASE}/api/training/export?batch_id=${batchId}`
      : `${API_BASE}/api/training/export`;
    const response = await fetchWithError(url);
    return response.json();
  },

  /**
   * Mark batch as exported.
   */
  async markExported(batchId: string): Promise<void> {
    await fetchWithError(`${API_BASE}/api/training/mark-exported/${batchId}`, {
      method: "POST",
    });
  },
};
