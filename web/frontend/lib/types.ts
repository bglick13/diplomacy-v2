/**
 * Core types for the Diplomacy web app.
 * Uses plain TypeScript types for UI layer, with Effect Schema for validation.
 */

// =============================================================================
// Power Types
// =============================================================================

export type Power =
  | "AUSTRIA"
  | "ENGLAND"
  | "FRANCE"
  | "GERMANY"
  | "ITALY"
  | "RUSSIA"
  | "TURKEY";

export const POWERS: Power[] = [
  "AUSTRIA",
  "ENGLAND",
  "FRANCE",
  "GERMANY",
  "ITALY",
  "RUSSIA",
  "TURKEY",
];

// =============================================================================
// Adapter / Bot Difficulty
// =============================================================================

export interface AdapterInfo {
  id: string | null;
  name: string;
  description: string;
}

// =============================================================================
// Board Context
// =============================================================================

export interface BoardContext {
  my_units: string[];
  my_centers: string[];
  opponent_units: Record<string, string[]>;
  opponent_centers: Record<string, string[]>;
  unowned_centers: string[];
  power_rankings: [string, number][];
  compact_map_view: string;
}

// =============================================================================
// Game State
// =============================================================================

export interface GameState {
  id: string;
  phase: string;
  year: number;
  is_done: boolean;
  human_power: string;
  board_context: BoardContext;
  valid_moves: Record<string, string[]>;
  all_units: Record<string, string[]>;
  all_centers: Record<string, string[]>;
  training_batch_id?: string | null;
  trajectories_collected?: number | null;
}

// =============================================================================
// API Request Types
// =============================================================================

export interface NewGameRequest {
  human_power: string;
  adapter_name: string | null;
  horizon?: number;
}

export interface SubmitOrdersRequest {
  orders: string[];
}

// =============================================================================
// API Response Types
// =============================================================================

export interface GameListItem {
  id: string;
  human_power: string;
  adapter_name: string | null;
  is_done: boolean;
  created_at: number;
  updated_at: number;
}

export interface TrainingStats {
  total_trajectories: number;
  exported_trajectories: number;
  unexported_trajectories: number;
  total_games: number;
  completed_games: number;
  batch_count: number;
  avg_trajectories_per_batch: number;
}

// =============================================================================
// Error Types
// =============================================================================

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

// =============================================================================
// Helper Types
// =============================================================================

/** Map of unit string to selected order */
export type OrderSelection = Record<string, string>;

/** Phase type extracted from phase string */
export type PhaseType = "M" | "R" | "A";

/**
 * Extract phase type from phase string like "SPRING 1901 MOVEMENT"
 */
export function getPhaseType(phase: string): PhaseType {
  if (phase.includes("MOVEMENT")) return "M";
  if (phase.includes("RETREAT")) return "R";
  if (phase.includes("ADJUSTMENT") || phase.includes("BUILD")) return "A";
  return "M"; // Default
}

/**
 * Get season from phase string
 */
export function getSeason(phase: string): "SPRING" | "FALL" | "WINTER" {
  if (phase.includes("SPRING")) return "SPRING";
  if (phase.includes("FALL")) return "FALL";
  return "WINTER";
}

/**
 * Power colors for UI
 */
export const POWER_COLORS: Record<Power, string> = {
  AUSTRIA: "#E53935", // Red
  ENGLAND: "#1E88E5", // Blue
  FRANCE: "#42A5F5", // Light blue
  GERMANY: "#6D6D6D", // Gray
  ITALY: "#4CAF50", // Green
  RUSSIA: "#9C27B0", // Purple
  TURKEY: "#FF9800", // Orange
};
