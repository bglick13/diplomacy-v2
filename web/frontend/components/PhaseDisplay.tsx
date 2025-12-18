"use client";

import { Badge } from "@/components/ui/badge";
import { getPhaseType, getSeason } from "@/lib/types";

interface PhaseDisplayProps {
  phase: string;
  year: number;
}

export function PhaseDisplay({ phase, year }: PhaseDisplayProps) {
  const phaseType = getPhaseType(phase);
  const season = getSeason(phase);

  const phaseTypeLabels: Record<string, string> = {
    M: "Movement",
    R: "Retreats",
    A: "Adjustments",
  };

  const seasonVariants: Record<string, string> = {
    SPRING: "bg-green-600 text-white",
    FALL: "bg-orange-600 text-white",
    WINTER: "bg-blue-600 text-white",
  };

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Badge className={seasonVariants[season]}>{season}</Badge>
        <span className="text-2xl font-bold">{year}</span>
      </div>
      <Badge variant="secondary">{phaseTypeLabels[phaseType]}</Badge>
    </div>
  );
}
