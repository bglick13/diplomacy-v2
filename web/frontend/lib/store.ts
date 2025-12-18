/**
 * Global state management using Zustand.
 */

import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { GameState, AdapterInfo, OrderSelection } from "./types";
import { POWERS } from "./types";

// =============================================================================
// Game Store
// =============================================================================

interface GameStore {
  // State
  game: GameState | null;
  selectedOrders: OrderSelection;
  isSubmitting: boolean;
  error: string | null;

  // Actions
  setGame: (game: GameState | null) => void;
  selectOrder: (unit: string, order: string) => void;
  clearOrders: () => void;
  removeOrder: (unit: string) => void;
  setSubmitting: (isSubmitting: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialGameState: Pick<
  GameStore,
  "game" | "selectedOrders" | "isSubmitting" | "error"
> = {
  game: null,
  selectedOrders: {},
  isSubmitting: false,
  error: null,
};

export const useGameStore = create<GameStore>()(
  devtools(
    (set) => ({
      ...initialGameState,

      setGame: (game) =>
        set({ game, selectedOrders: {}, error: null }, false, "setGame"),

      selectOrder: (unit, order) =>
        set(
          (state) => ({
            selectedOrders: { ...state.selectedOrders, [unit]: order },
          }),
          false,
          "selectOrder"
        ),

      clearOrders: () => set({ selectedOrders: {} }, false, "clearOrders"),

      removeOrder: (unit) =>
        set(
          (state) => {
            const { [unit]: _, ...rest } = state.selectedOrders;
            return { selectedOrders: rest };
          },
          false,
          "removeOrder"
        ),

      setSubmitting: (isSubmitting) =>
        set({ isSubmitting }, false, "setSubmitting"),

      setError: (error) => set({ error }, false, "setError"),

      reset: () => set(initialGameState, false, "reset"),
    }),
    { name: "game-store" }
  )
);

// =============================================================================
// Game Creation Store
// =============================================================================

interface GameCreationStore {
  // State
  selectedPower: string;
  selectedAdapter: string | null;
  availableAdapters: AdapterInfo[];
  isCreating: boolean;
  error: string | null;

  // Actions
  setSelectedPower: (power: string) => void;
  setSelectedAdapter: (adapterId: string | null) => void;
  setAvailableAdapters: (adapters: AdapterInfo[]) => void;
  setCreating: (isCreating: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialCreationState: Pick<
  GameCreationStore,
  "selectedPower" | "selectedAdapter" | "availableAdapters" | "isCreating" | "error"
> = {
  selectedPower: "FRANCE",
  selectedAdapter: null,
  availableAdapters: [],
  isCreating: false,
  error: null,
};

export const useGameCreationStore = create<GameCreationStore>()(
  devtools(
    (set) => ({
      ...initialCreationState,

      setSelectedPower: (selectedPower) =>
        set({ selectedPower }, false, "setSelectedPower"),

      setSelectedAdapter: (selectedAdapter) =>
        set({ selectedAdapter }, false, "setSelectedAdapter"),

      setAvailableAdapters: (availableAdapters) =>
        set({ availableAdapters }, false, "setAvailableAdapters"),

      setCreating: (isCreating) => set({ isCreating }, false, "setCreating"),

      setError: (error) => set({ error }, false, "setError"),

      reset: () => set(initialCreationState, false, "reset"),
    }),
    { name: "game-creation-store" }
  )
);

// =============================================================================
// Selectors / Derived State
// =============================================================================

/**
 * Check if all units have orders selected.
 */
export function useAllOrdersSelected(): boolean {
  const game = useGameStore((state) => state.game);
  const selectedOrders = useGameStore((state) => state.selectedOrders);

  if (!game) return false;

  const requiredUnits = Object.keys(game.valid_moves);
  return requiredUnits.every((unit) => selectedOrders[unit] !== undefined);
}

/**
 * Get the count of selected orders vs required.
 */
export function useOrdersProgress(): { selected: number; required: number } {
  const game = useGameStore((state) => state.game);
  const selectedOrders = useGameStore((state) => state.selectedOrders);

  if (!game) return { selected: 0, required: 0 };

  const required = Object.keys(game.valid_moves).length;
  const selected = Object.keys(selectedOrders).length;

  return { selected, required };
}

/**
 * Get orders as array for submission.
 */
export function useOrdersArray(): string[] {
  const selectedOrders = useGameStore((state) => state.selectedOrders);
  return Object.values(selectedOrders);
}
