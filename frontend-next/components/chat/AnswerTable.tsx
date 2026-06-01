"use client";

import { useState } from "react";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { toast } from "sonner";

interface AnswerTableProps {
  headers: string[];
  rows: string[][];
}

type SortDir = "asc" | "desc" | null;

function isNumeric(v: string) {
  return v.trim() !== "" && !isNaN(Number(v.replace(/[,$%]/g, "")));
}

export function AnswerTable({ headers, rows }: AnswerTableProps) {
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>(null);

  function toggleSort(col: number) {
    if (sortCol !== col) { setSortCol(col); setSortDir("asc"); return; }
    if (sortDir === "asc") { setSortDir("desc"); return; }
    setSortCol(null); setSortDir(null);
  }

  const sorted = sortCol == null ? rows : [...rows].sort((a, b) => {
    const av = a[sortCol] ?? "";
    const bv = b[sortCol] ?? "";
    const aNum = Number(av.replace(/[,$%]/g, ""));
    const bNum = Number(bv.replace(/[,$%]/g, ""));
    const numeric = isNumeric(av) && isNumeric(bv);
    const cmp = numeric ? aNum - bNum : av.localeCompare(bv);
    return sortDir === "desc" ? -cmp : cmp;
  });

  // Detect numeric columns for right-align
  const numericCols = new Set(
    headers.map((_, ci) => rows.every((r) => isNumeric(r[ci] ?? "")) ? ci : -1).filter((v) => v >= 0)
  );

  return (
    <div className="my-3 rounded-xl border border-[var(--border)] overflow-hidden">
      <div className="overflow-x-auto scrollbar-thin">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--bg-base)]">
              {headers.map((h, ci) => (
                <th
                  key={ci}
                  onClick={() => toggleSort(ci)}
                  className={`px-3 py-2.5 font-semibold text-[var(--text-primary)] whitespace-nowrap cursor-pointer select-none group
                    ${numericCols.has(ci) ? "text-right" : "text-left"}`}
                >
                  <span className="inline-flex items-center gap-1">
                    {h}
                    <span className="text-[var(--text-muted)] opacity-0 group-hover:opacity-100 transition-opacity">
                      {sortCol === ci ? (
                        sortDir === "asc" ? <ArrowUp size={10} /> : <ArrowDown size={10} />
                      ) : (
                        <ArrowUpDown size={10} />
                      )}
                    </span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, ri) => (
              <tr
                key={ri}
                className={`border-b border-[var(--border)] last:border-0 transition-colors hover:bg-[var(--bg-hover)]`}
              >
                {headers.map((_, ci) => (
                  <td
                    key={ci}
                    className={`px-3 py-2 text-[var(--text-secondary)] leading-snug
                      ${numericCols.has(ci) ? "text-right font-mono text-[11px]" : "text-left"}`}
                  >
                    {row[ci] ?? "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Export stub */}
      <div className="px-3 py-2 border-t border-[var(--border)] flex items-center justify-end bg-[var(--bg-base)]">
        <button
          onClick={() => toast.info("XLSX export coming soon")}
          className="text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
        >
          Export ↓
        </button>
      </div>
    </div>
  );
}
