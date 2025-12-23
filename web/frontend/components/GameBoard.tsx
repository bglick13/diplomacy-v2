"use client";

import { useCallback } from "react";
import { DiplomacyMap } from "./DiplomacyMap";
import { useDragOrderState } from "@/hooks/useDragOrderState";
import { useGameStore } from "@/lib/store";

interface GameBoardProps {
  units: Record<string, string[]>;
  centers: Record<string, string[]>;
  humanPower: string;
  compactMapView?: string;
  validMoves?: Record<string, string[]>;
  selectedOrders?: Record<string, string>;
  selectedUnit?: string | null;
  onUnitClick?: (unit: string) => void;
}

/**
 * Game board visualization using the SVG map with drag-and-drop order input.
 */
export function GameBoard({
  units,
  centers,
  humanPower,
  validMoves = {},
  selectedOrders,
  selectedUnit,
  onUnitClick,
}: GameBoardProps) {
  const selectOrder = useGameStore((state) => state.selectOrder);

  // Handle order completion from drag-and-drop
  const handleOrderComplete = useCallback(
    (unit: string, order: string) => {
      selectOrder(unit, order);
    },
    [selectOrder]
  );

  // Initialize drag state hook
  const {
    state: dragState,
    startDrag,
    updateDrag,
    endDrag,
    cancelDrag,
  } = useDragOrderState({
    validMoves,
    selectedOrders: selectedOrders || {},
    onOrderComplete: handleOrderComplete,
  });

  // Click handler - only for external click handling (e.g., selection)
  // Hold orders are now handled by pointer events (click = pointerdown + pointerup without drag)
  const handleCombinedUnitClick = useCallback(
    (unit: string) => {
      // If we're in a multi-step drag state, don't trigger click
      if (dragState.phase !== "idle") return;

      // Call original click handler if provided (for selection, etc.)
      onUnitClick?.(unit);
    },
    [dragState.phase, onUnitClick]
  );

  return (
    <DiplomacyMap
      units={units}
      centers={centers}
      humanPower={humanPower}
      validMoves={validMoves}
      selectedOrders={selectedOrders}
      selectedUnit={selectedUnit}
      onUnitClick={handleCombinedUnitClick}
      dragState={dragState}
      onDragStart={startDrag}
      onDragMove={updateDrag}
      onDragEnd={endDrag}
      onDragCancel={cancelDrag}
    />
  );
}
