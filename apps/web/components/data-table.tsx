"use client";

import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp, Download, Search } from "lucide-react";
import { useMemo, useState } from "react";

import type { Row, Rows } from "@/lib/contracts";

interface DataTableProps {
  rows: Rows;
  columns?: string[];
  emptyTitle: string;
  emptyDetail: string;
  maxRows?: number;
  exportName?: string;
}

function formatCell(value: Row[string]): string {
  if (value === null || value === undefined || value === "") return "Unavailable";
  if (typeof value === "number") {
    if (Math.abs(value) < 1 && value !== 0) return value.toFixed(4);
    return new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(value);
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function sortableCell(value: Row[string]): number | string {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (value === null || value === undefined) return "";
  return String(value).toLocaleLowerCase();
}

function csvCell(value: Row[string]): string {
  const text = value === null || value === undefined ? "" : typeof value === "object" ? JSON.stringify(value) : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

export function DataTable({
  rows,
  columns,
  emptyTitle,
  emptyDetail,
  maxRows = 40,
  exportName = "terminal-data",
}: DataTableProps) {
  const [query, setQuery] = useState("");
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(0);
  const visibleColumns = useMemo(
    () => (rows.length ? columns?.filter((column) => column in rows[0]) ?? Object.keys(rows[0]).slice(0, 9) : columns ?? []),
    [columns, rows],
  );
  const pageSize = Math.min(15, maxRows);
  const filteredRows = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const filtered = needle
      ? rows.filter((row) => visibleColumns.some((column) => formatCell(row[column]).toLocaleLowerCase().includes(needle)))
      : [...rows];
    if (sortColumn) {
      filtered.sort((left, right) => {
        const a = sortableCell(left[sortColumn]);
        const b = sortableCell(right[sortColumn]);
        const order = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
        return sortDirection === "asc" ? order : -order;
      });
    }
    return filtered;
  }, [query, rows, sortColumn, sortDirection, visibleColumns]);
  const boundedRows = filteredRows.slice(0, maxRows);
  const pageCount = Math.max(1, Math.ceil(boundedRows.length / pageSize));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = boundedRows.slice(safePage * pageSize, (safePage + 1) * pageSize);

  function setSort(column: string) {
    setPage(0);
    if (sortColumn === column) setSortDirection((value) => (value === "asc" ? "desc" : "asc"));
    else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  }

  function exportCsv() {
    const csv = [visibleColumns.map(csvCell).join(","), ...filteredRows.map((row) => visibleColumns.map((column) => csvCell(row[column])).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${exportName}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  if (!rows.length) {
    return (
      <div className="empty-state" role="status">
        <strong>{emptyTitle}</strong>
        <span>{emptyDetail}</span>
      </div>
    );
  }

  return (
    <div className="table-frame">
      <div className="table-toolbar">
        <label>
          <Search size={15} aria-hidden="true" />
          <span className="sr-only">Filter table</span>
          <input
            type="search"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
            placeholder="Filter records"
          />
        </label>
        <span>{filteredRows.length.toLocaleString("en-US")} records</span>
        <button type="button" onClick={exportCsv} aria-label="Export filtered table to CSV">
          <Download size={15} aria-hidden="true" />
          Export CSV
        </button>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            {visibleColumns.map((column) => (
              <th key={column} scope="col">
                <button type="button" onClick={() => setSort(column)} aria-label={`Sort by ${column.replaceAll("_", " ")}`}>
                  <span>{column.replaceAll("_", " ")}</span>
                  {sortColumn === column ? (
                    sortDirection === "asc" ? <ChevronUp size={13} aria-hidden="true" /> : <ChevronDown size={13} aria-hidden="true" />
                  ) : null}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pageRows.map((row, index) => (
            <tr key={`${String(row.Ticker ?? row.Metric ?? row.Country ?? "row")}-${index}`}>
              {visibleColumns.map((column) => (
                <td key={column}>{formatCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="table-pagination">
        <span>
          {boundedRows.length ? safePage * pageSize + 1 : 0}–{Math.min((safePage + 1) * pageSize, boundedRows.length)} of {boundedRows.length}
          {filteredRows.length > maxRows ? ` (capped from ${filteredRows.length})` : ""}
        </span>
        <div>
          <button type="button" aria-label="Previous table page" disabled={safePage === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
            <ChevronLeft size={16} aria-hidden="true" />
          </button>
          <span>{safePage + 1} / {pageCount}</span>
          <button type="button" aria-label="Next table page" disabled={safePage >= pageCount - 1} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>
            <ChevronRight size={16} aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  );
}
