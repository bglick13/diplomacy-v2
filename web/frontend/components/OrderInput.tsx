"use client";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  useGameStore,
  useAllOrdersSelected,
  useOrdersProgress,
} from "@/lib/store";

interface OrderInputProps {
  validMoves: Record<string, string[]>;
  onSubmit: () => void;
}

export function OrderInput({ validMoves, onSubmit }: OrderInputProps) {
  const { selectedOrders, selectOrder, isSubmitting } = useGameStore();
  const allSelected = useAllOrdersSelected();
  const { selected, required } = useOrdersProgress();

  const units = Object.entries(validMoves);

  if (units.length === 0) {
    return (
      <div className="space-y-3">
        <p className="text-muted-foreground text-center text-sm">
          No orders required this phase.
        </p>
        <Button onClick={onSubmit} className="w-full" size="sm" disabled={isSubmitting}>
          {isSubmitting ? "Processing..." : "Continue"}
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Orders</span>
        <Badge variant={allSelected ? "default" : "secondary"} className="text-xs">
          {selected}/{required}
        </Badge>
      </div>

      <div className="space-y-2">
        {units.map(([unit, options]) => {
          const selectedOrder = selectedOrders[unit];
          const isSelected = selectedOrder !== undefined;

          return (
            <div key={unit} className="space-y-1">
              <label className="text-xs font-medium flex items-center gap-1.5">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isSelected ? "bg-green-500" : "bg-secondary-foreground/30"
                  }`}
                />
                {unit}
              </label>
              <Select
                value={selectedOrder ?? ""}
                onValueChange={(value) => selectOrder(unit, value)}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue placeholder="Select..." />
                </SelectTrigger>
                <SelectContent className="max-h-60">
                  {options.map((order) => {
                    const orderType = getOrderType(order);

                    return (
                      <SelectItem key={order} value={order}>
                        <div className="flex items-center gap-1.5">
                          <Badge
                            variant="outline"
                            className={`text-[10px] px-1 py-0 ${getOrderColor(orderType)}`}
                          >
                            {orderType}
                          </Badge>
                          <span className="font-mono text-xs">{order}</span>
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
          );
        })}
      </div>

      <Button
        onClick={onSubmit}
        disabled={!allSelected || isSubmitting}
        className="w-full"
        size="sm"
      >
        {isSubmitting
          ? "Processing..."
          : allSelected
            ? "Submit Orders"
            : `Select ${required - selected} more`}
      </Button>
    </div>
  );
}

function getOrderType(order: string): string {
  if (order.includes(" - ")) return "Move";
  if (order.includes(" S ")) return "Support";
  if (order.includes(" C ")) return "Convoy";
  if (order.includes(" H")) return "Hold";
  if (order.includes(" B")) return "Build";
  if (order.includes(" D")) return "Disband";
  if (order === "WAIVE") return "Waive";
  return "Order";
}

function getOrderColor(type: string): string {
  switch (type) {
    case "Move":
      return "border-blue-500 text-blue-500";
    case "Support":
      return "border-green-500 text-green-500";
    case "Convoy":
      return "border-purple-500 text-purple-500";
    case "Hold":
      return "border-muted-foreground text-muted-foreground";
    case "Build":
      return "border-emerald-500 text-emerald-500";
    case "Disband":
      return "border-red-500 text-red-500";
    default:
      return "border-muted-foreground text-muted-foreground";
  }
}
