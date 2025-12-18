"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { POWER_COLORS, type Power } from "@/lib/types";

interface PowerStatusProps {
  rankings: [string, number][];
  humanPower: string;
  allUnits?: Record<string, string[]>;
}

export function PowerStatus({
  rankings,
  humanPower,
  allUnits,
}: PowerStatusProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Power Rankings</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          {rankings.map(([power, centers], index) => {
            const isHuman = power === humanPower;
            const unitCount = allUnits?.[power]?.length ?? 0;

            return (
              <div
                key={power}
                className={`flex items-center justify-between p-2 rounded ${
                  isHuman ? "bg-muted" : ""
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-sm w-4">
                    {index + 1}.
                  </span>
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: POWER_COLORS[power as Power] }}
                  />
                  <span className={isHuman ? "font-medium" : ""}>
                    {power}
                    {isHuman && (
                      <span className="text-xs text-muted-foreground ml-1">
                        (You)
                      </span>
                    )}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-muted-foreground">{unitCount} units</span>
                  <span className="font-medium">
                    {centers} SC{centers !== 1 ? "s" : ""}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
