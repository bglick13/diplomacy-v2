/**
 * Drag state machine hook for drag-and-drop order input.
 *
 * State machine phases:
 * IDLE → DRAGGING → MOVE (complete)
 *               ↘ SELECTING_SUPPORT_TARGET → SELECTING_SUPPORT_DEST → SUPPORT (complete)
 *               ↘ SELECTING_CONVOY_ARMY → SELECTING_CONVOY_DEST → CONVOY (complete)
 */

import { useReducer, useCallback, useEffect } from "react";
import {
  getMoveDestinations,
  getSupportableLocations,
  getSupportMoveDestinations,
  getConvoyableArmyLocations,
  getConvoyDestinations,
  getHoldOrder,
  findMatchingOrder,
  getUnitTypeAtLocation,
  getBaseProvince,
} from "@/lib/order-utils";

export interface Point {
  x: number;
  y: number;
}

// State types
interface IdleState {
  phase: "idle";
}

interface DraggingState {
  phase: "dragging";
  unit: string;
  startPos: Point;
  currentPos: Point;
  validMoveDestinations: string[];
  validSupportLocations: string[];
  validConvoyArmyLocations: string[];
  canHold: boolean;
}

interface SelectingSupportTargetState {
  phase: "selecting_support_target";
  unit: string;
  validTargetLocations: string[];
}

interface SelectingSupportDestState {
  phase: "selecting_support_dest";
  unit: string;
  targetLocation: string;
  targetUnit: string;
  validDestinations: string[];
}

interface SelectingConvoyArmyState {
  phase: "selecting_convoy_army";
  unit: string;
  validArmyLocations: string[];
}

interface SelectingConvoyDestState {
  phase: "selecting_convoy_dest";
  unit: string;
  armyLocation: string;
  armyUnit: string;
  validDestinations: string[];
}

export type DragState =
  | IdleState
  | DraggingState
  | SelectingSupportTargetState
  | SelectingSupportDestState
  | SelectingConvoyArmyState
  | SelectingConvoyDestState;

// Actions
type DragAction =
  | { type: "START_DRAG"; unit: string; pos: Point; validMoves: string[] }
  | { type: "UPDATE_DRAG"; pos: Point }
  | { type: "END_DRAG_ON_MOVE"; destination: string; validMoves: string[] }
  | { type: "END_DRAG_ON_HOLD"; validMoves: string[] }
  | { type: "END_DRAG_ON_SUPPORT_TARGET"; targetLocation: string; validMoves: string[] }
  | { type: "END_DRAG_ON_CONVOY_ARMY"; armyLocation: string; validMoves: string[] }
  | { type: "SELECT_SUPPORT_DEST"; destination: string; validMoves: string[] }
  | { type: "SELECT_CONVOY_DEST"; destination: string; validMoves: string[] }
  | { type: "ENTER_SUPPORT_MODE"; validMoves: string[] }
  | { type: "ENTER_CONVOY_MODE"; validMoves: string[] }
  | { type: "CANCEL" }
  | { type: "UNDO" };

// Result types for completed orders
export interface DragOrderResult {
  unit: string;
  order: string;
}

function reducer(state: DragState, action: DragAction): DragState {
  switch (action.type) {
    case "START_DRAG": {
      const validMoves = action.validMoves;
      return {
        phase: "dragging",
        unit: action.unit,
        startPos: action.pos,
        currentPos: action.pos,
        validMoveDestinations: getMoveDestinations(validMoves),
        validSupportLocations: getSupportableLocations(validMoves),
        validConvoyArmyLocations: getConvoyableArmyLocations(validMoves),
        canHold: getHoldOrder(validMoves) !== null,
      };
    }

    case "UPDATE_DRAG": {
      if (state.phase !== "dragging") return state;
      return { ...state, currentPos: action.pos };
    }

    case "END_DRAG_ON_MOVE": {
      // Complete move order - handled externally
      return { phase: "idle" };
    }

    case "END_DRAG_ON_HOLD": {
      // Complete hold order - handled externally
      return { phase: "idle" };
    }

    case "END_DRAG_ON_SUPPORT_TARGET": {
      if (state.phase !== "dragging" && state.phase !== "selecting_support_target") {
        return state;
      }

      const unit = state.unit;
      const validMoves = action.validMoves;
      const targetLocation = action.targetLocation;
      const validDestinations = getSupportMoveDestinations(validMoves, targetLocation);

      // Get the unit type at target location
      const targetUnitType = getUnitTypeAtLocation(validMoves, targetLocation);
      const targetUnit = targetUnitType ? `${targetUnitType} ${targetLocation}` : `A ${targetLocation}`;

      return {
        phase: "selecting_support_dest",
        unit,
        targetLocation,
        targetUnit,
        validDestinations,
      };
    }

    case "END_DRAG_ON_CONVOY_ARMY": {
      if (state.phase !== "dragging" && state.phase !== "selecting_convoy_army") {
        return state;
      }

      const unit = state.unit;
      const validMoves = action.validMoves;
      const armyLocation = action.armyLocation;
      const validDestinations = getConvoyDestinations(validMoves, armyLocation);
      const armyUnit = `A ${armyLocation}`;

      return {
        phase: "selecting_convoy_dest",
        unit,
        armyLocation,
        armyUnit,
        validDestinations,
      };
    }

    case "SELECT_SUPPORT_DEST": {
      // Complete support order - handled externally
      return { phase: "idle" };
    }

    case "SELECT_CONVOY_DEST": {
      // Complete convoy order - handled externally
      return { phase: "idle" };
    }

    case "ENTER_SUPPORT_MODE": {
      if (state.phase !== "dragging") return state;
      return {
        phase: "selecting_support_target",
        unit: state.unit,
        validTargetLocations: state.validSupportLocations,
      };
    }

    case "ENTER_CONVOY_MODE": {
      if (state.phase !== "dragging") return state;
      return {
        phase: "selecting_convoy_army",
        unit: state.unit,
        validArmyLocations: state.validConvoyArmyLocations,
      };
    }

    case "CANCEL": {
      return { phase: "idle" };
    }

    case "UNDO": {
      // Go back one step
      if (state.phase === "selecting_support_dest") {
        return {
          phase: "selecting_support_target",
          unit: state.unit,
          validTargetLocations: [], // Will need to recalculate
        };
      }
      if (state.phase === "selecting_convoy_dest") {
        return {
          phase: "selecting_convoy_army",
          unit: state.unit,
          validArmyLocations: [], // Will need to recalculate
        };
      }
      return { phase: "idle" };
    }

    default:
      return state;
  }
}

export interface UseDragOrderStateOptions {
  validMoves: Record<string, string[]>;
  selectedOrders: Record<string, string>;
  onOrderComplete: (unit: string, order: string) => void;
}

export function useDragOrderState({ validMoves, selectedOrders, onOrderComplete }: UseDragOrderStateOptions) {
  const [state, dispatch] = useReducer(reducer, { phase: "idle" });

  // Helper: Find if any selected order is moving TO a given province
  const findMoveToProvince = useCallback(
    (targetProvince: string): { unit: string; order: string } | null => {
      const baseTarget = getBaseProvince(targetProvince);
      for (const [unit, order] of Object.entries(selectedOrders)) {
        // Check if this is a move order to the target
        const moveMatch = order.match(/- ([A-Z]{3}(?:\/[A-Z]{2})?)$/);
        if (moveMatch && getBaseProvince(moveMatch[1]) === baseTarget) {
          return { unit, order };
        }
      }
      return null;
    },
    [selectedOrders]
  );

  // Start dragging a unit
  const startDrag = useCallback(
    (unit: string, pos: Point) => {
      const unitMoves = validMoves[unit];
      if (!unitMoves || unitMoves.length === 0) return;

      dispatch({ type: "START_DRAG", unit, pos, validMoves: unitMoves });
    },
    [validMoves]
  );

  // Update cursor position during drag
  const updateDrag = useCallback((pos: Point) => {
    dispatch({ type: "UPDATE_DRAG", pos });
  }, []);

  // Handle dropping on a province/location
  const endDrag = useCallback(
    (targetProvince: string | null) => {
      if (state.phase === "idle") return;

      const unit = state.unit;
      const unitMoves = validMoves[unit] || [];

      // Dragging phase - determine what type of order
      if (state.phase === "dragging") {
        // No target = clicked without dragging = hold order
        if (!targetProvince) {
          const holdOrder = getHoldOrder(unitMoves);
          if (holdOrder) {
            onOrderComplete(unit, holdOrder);
          }
          dispatch({ type: "END_DRAG_ON_HOLD", validMoves: unitMoves });
          return;
        }

        const baseTarget = getBaseProvince(targetProvince);
        const unitLocation = unit.split(" ")[1];
        const baseUnitLocation = getBaseProvince(unitLocation);

        // Dropped on self = hold
        if (baseTarget === baseUnitLocation) {
          const holdOrder = getHoldOrder(unitMoves);
          if (holdOrder) {
            onOrderComplete(unit, holdOrder);
          }
          dispatch({ type: "END_DRAG_ON_HOLD", validMoves: unitMoves });
          return;
        }

        // Check if there's already a move order to this province - if so, try to support it
        const existingMoveToTarget = findMoveToProvince(targetProvince);
        if (existingMoveToTarget) {
          // Find the location of the unit making that move
          const movingUnitLoc = existingMoveToTarget.unit.split(" ")[1];
          const baseMovingUnitLoc = getBaseProvince(movingUnitLoc);

          // Check if we can support that unit's move to this destination
          if (state.validSupportLocations.some((loc) => getBaseProvince(loc) === baseMovingUnitLoc)) {
            // Try to find a support-move order for this exact combination
            const supportOrder = findMatchingOrder(unitMoves, unit, {
              type: "support_move",
              targetLocation: baseMovingUnitLoc,
              targetTo: baseTarget,
            });
            if (supportOrder) {
              onOrderComplete(unit, supportOrder);
              dispatch({ type: "END_DRAG_ON_MOVE", destination: targetProvince, validMoves: unitMoves });
              return;
            }
          }
        }

        // Check if it's a valid move destination
        if (state.validMoveDestinations.some((d) => getBaseProvince(d) === baseTarget)) {
          const order = findMatchingOrder(unitMoves, unit, { type: "move", to: targetProvince });
          // Also try base province if exact match not found
          const orderAlt = order || findMatchingOrder(unitMoves, unit, { type: "move", to: baseTarget });
          // Try coastal variants
          const finalOrder = orderAlt || unitMoves.find((o) => o.includes(` - ${baseTarget}`));
          if (finalOrder) {
            onOrderComplete(unit, finalOrder);
          }
          dispatch({ type: "END_DRAG_ON_MOVE", destination: targetProvince, validMoves: unitMoves });
          return;
        }

        // Check if it's a unit to support
        if (state.validSupportLocations.some((loc) => getBaseProvince(loc) === baseTarget)) {
          dispatch({ type: "END_DRAG_ON_SUPPORT_TARGET", targetLocation: baseTarget, validMoves: unitMoves });
          return;
        }

        // Check if it's an army to convoy
        if (state.validConvoyArmyLocations.some((loc) => getBaseProvince(loc) === baseTarget)) {
          dispatch({ type: "END_DRAG_ON_CONVOY_ARMY", armyLocation: baseTarget, validMoves: unitMoves });
          return;
        }

        // Invalid drop
        dispatch({ type: "CANCEL" });
        return;
      }

      // Selecting support destination
      if (state.phase === "selecting_support_dest") {
        if (!targetProvince) {
          dispatch({ type: "CANCEL" });
          return;
        }

        const baseTarget = getBaseProvince(targetProvince);

        // Validate destination
        if (state.validDestinations.some((d) => getBaseProvince(d) === baseTarget)) {
          // Check if it's support-hold or support-move
          if (baseTarget === state.targetLocation) {
            // Support hold
            const order = findMatchingOrder(unitMoves, unit, {
              type: "support_hold",
              targetLocation: state.targetLocation,
            });
            if (order) {
              onOrderComplete(unit, order);
            }
          } else {
            // Support move
            const order = findMatchingOrder(unitMoves, unit, {
              type: "support_move",
              targetLocation: state.targetLocation,
              targetTo: baseTarget,
            });
            // Try with exact targetProvince too
            const orderAlt = order || findMatchingOrder(unitMoves, unit, {
              type: "support_move",
              targetLocation: state.targetLocation,
              targetTo: targetProvince,
            });
            if (orderAlt) {
              onOrderComplete(unit, orderAlt);
            }
          }
          dispatch({ type: "SELECT_SUPPORT_DEST", destination: targetProvince, validMoves: unitMoves });
          return;
        }

        dispatch({ type: "CANCEL" });
        return;
      }

      // Selecting convoy destination
      if (state.phase === "selecting_convoy_dest") {
        if (!targetProvince) {
          dispatch({ type: "CANCEL" });
          return;
        }

        const baseTarget = getBaseProvince(targetProvince);

        if (state.validDestinations.some((d) => getBaseProvince(d) === baseTarget)) {
          const order = findMatchingOrder(unitMoves, unit, {
            type: "convoy",
            armyLocation: state.armyLocation,
            to: baseTarget,
          });
          const orderAlt = order || findMatchingOrder(unitMoves, unit, {
            type: "convoy",
            armyLocation: state.armyLocation,
            to: targetProvince,
          });
          if (orderAlt) {
            onOrderComplete(unit, orderAlt);
          }
          dispatch({ type: "SELECT_CONVOY_DEST", destination: targetProvince, validMoves: unitMoves });
          return;
        }

        dispatch({ type: "CANCEL" });
        return;
      }

      // Selecting support target (from support mode button)
      if (state.phase === "selecting_support_target") {
        if (!targetProvince) {
          dispatch({ type: "CANCEL" });
          return;
        }

        const baseTarget = getBaseProvince(targetProvince);
        if (state.validTargetLocations.some((loc) => getBaseProvince(loc) === baseTarget)) {
          dispatch({ type: "END_DRAG_ON_SUPPORT_TARGET", targetLocation: baseTarget, validMoves: unitMoves });
          return;
        }

        dispatch({ type: "CANCEL" });
        return;
      }

      // Selecting convoy army (from convoy mode button)
      if (state.phase === "selecting_convoy_army") {
        if (!targetProvince) {
          dispatch({ type: "CANCEL" });
          return;
        }

        const baseTarget = getBaseProvince(targetProvince);
        if (state.validArmyLocations.some((loc) => getBaseProvince(loc) === baseTarget)) {
          dispatch({ type: "END_DRAG_ON_CONVOY_ARMY", armyLocation: baseTarget, validMoves: unitMoves });
          return;
        }

        dispatch({ type: "CANCEL" });
        return;
      }
    },
    [state, validMoves, onOrderComplete]
  );

  // Cancel drag operation
  const cancelDrag = useCallback(() => {
    dispatch({ type: "CANCEL" });
  }, []);

  // Handle click (tap) on a unit - for hold order
  const handleUnitClick = useCallback(
    (unit: string) => {
      if (state.phase !== "idle") return;

      const unitMoves = validMoves[unit];
      if (!unitMoves) return;

      const holdOrder = getHoldOrder(unitMoves);
      if (holdOrder) {
        onOrderComplete(unit, holdOrder);
      }
    },
    [state.phase, validMoves, onOrderComplete]
  );

  // Enter support mode (for button trigger)
  const enterSupportMode = useCallback(() => {
    if (state.phase !== "dragging") return;
    dispatch({ type: "ENTER_SUPPORT_MODE", validMoves: validMoves[state.unit] || [] });
  }, [state, validMoves]);

  // Enter convoy mode (for button trigger)
  const enterConvoyMode = useCallback(() => {
    if (state.phase !== "dragging") return;
    dispatch({ type: "ENTER_CONVOY_MODE", validMoves: validMoves[state.unit] || [] });
  }, [state, validMoves]);

  // Undo last step
  const undoStep = useCallback(() => {
    dispatch({ type: "UNDO" });
  }, []);

  // Handle ESC key to cancel
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        dispatch({ type: "CANCEL" });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return {
    state,
    startDrag,
    updateDrag,
    endDrag,
    cancelDrag,
    handleUnitClick,
    enterSupportMode,
    enterConvoyMode,
    undoStep,
    isDragging: state.phase !== "idle",
  };
}
