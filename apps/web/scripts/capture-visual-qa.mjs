import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "../../..");
const OUT_DIR = resolve(ROOT, "research_artifacts", "visual_qa");
const PORT = Number(process.env.QPK_VISUAL_QA_PORT ?? 3187);
const HOST = "127.0.0.1";
const BASE_URL = process.env.QPK_VISUAL_QA_URL ?? `http://${HOST}:${PORT}`;
const SCREENSHOT_TIMEOUT_MS = Number(process.env.QPK_VISUAL_QA_SCREENSHOT_TIMEOUT_MS ?? 30_000);

const ROUTES = [
  { path: "/", name: "command-center" },
  { path: "/market-intelligence", name: "market-intelligence" },
  { path: "/rates-fixed-income", name: "rates-fixed-income" },
  { path: "/equity-research", name: "equity-research" },
  { path: "/xcdr-research", name: "xcdr-research" },
  { path: "/risk-laboratory", name: "risk-laboratory" },
  { path: "/data-quality", name: "data-quality" },
];

const VIEWPORTS = [
  { name: "mobile-375", width: 375, height: 900 },
  { name: "tablet-768", width: 768, height: 1024 },
  { name: "laptop-1440", width: 1440, height: 1100 },
  { name: "desktop-1920", width: 1920, height: 1200 },
];

function requestedSet(name) {
  const raw = process.env[name];
  if (!raw) return null;
  return new Set(
    raw
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
}

function npmCommand() {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}

function startCommand() {
  const args = ["run", "start", "--workspace", "@qpk/web", "--", "--hostname", HOST, "--port", String(PORT)];
  if (process.platform === "win32") {
    return {
      command: "cmd.exe",
      args: ["/d", "/s", "/c", `${npmCommand()} ${args.join(" ")}`],
    };
  }
  return { command: npmCommand(), args };
}

function chromeCandidates() {
  const candidates = [];
  if (process.env.CHROME_PATH) candidates.push(process.env.CHROME_PATH);
  if (process.platform === "win32") {
    candidates.push(
      "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
      "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
      "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    );
  } else if (process.platform === "darwin") {
    candidates.push(
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
    );
  } else {
    candidates.push("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge");
  }
  return candidates.find((candidate) => existsSync(candidate));
}

async function waitForServer(url, timeoutMs = 45_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = "not attempted";
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (response.ok || response.status === 401 || response.status === 403) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 750));
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError}`);
}

function stopProcessTree(child) {
  if (!child?.pid) return;
  if (process.platform === "win32") {
    spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
  } else {
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      child.kill("SIGTERM");
    }
  }
}

function sleepSync(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function waitForFileBytes(file, minBytes = 5_000, timeoutMs = 5_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (existsSync(file)) {
      const bytes = statSync(file).size;
      if (bytes >= minBytes) return bytes;
    }
    sleepSync(100);
  }
  return existsSync(file) ? statSync(file).size : 0;
}
function safeName(value) {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-|-$/g, "").toLowerCase();
}

async function main() {
  mkdirSync(OUT_DIR, { recursive: true });
  const chrome = chromeCandidates();
  if (!chrome) {
    throw new Error("No Chrome/Edge executable found. Set CHROME_PATH to run visual QA screenshots.");
  }

  const routeFilter = requestedSet("QPK_VISUAL_QA_ROUTES");
  const viewportFilter = requestedSet("QPK_VISUAL_QA_VIEWPORTS");
  const selectedRoutes = routeFilter ? ROUTES.filter((route) => routeFilter.has(route.name) || routeFilter.has(route.path)) : ROUTES;
  const selectedViewports = viewportFilter ? VIEWPORTS.filter((viewport) => viewportFilter.has(viewport.name)) : VIEWPORTS;
  const start = startCommand();
  const server = spawn(
    start.command,
    start.args,
    {
      cwd: ROOT,
      env: {
        ...process.env,
        NODE_ENV: "production",
        QPK_ALLOW_LOCAL_ARTIFACTS: process.env.QPK_ALLOW_LOCAL_ARTIFACTS ?? "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    },
  );
  const serverLog = [];
  server.stdout?.on("data", (chunk) => serverLog.push(String(chunk)));
  server.stderr?.on("data", (chunk) => serverLog.push(String(chunk)));

  const report = {
    generated_at: new Date().toISOString(),
    base_url: BASE_URL,
    browser: chrome,
    routes: selectedRoutes.map((route) => route.path),
    viewports: selectedViewports,
    captures: [],
    warnings: [],
  };

  try {
    await waitForServer(`${BASE_URL}/`);
    for (const route of selectedRoutes) {
      for (const viewport of selectedViewports) {
        const file = join(OUT_DIR, `${safeName(route.name)}__${viewport.name}.png`);
        const url = `${BASE_URL}${route.path}`;
        try {
          await waitForServer(url, 20_000);
        } catch (error) {
          report.warnings.push({
            route: route.path,
            viewport: viewport.name,
            status: "unreachable",
            stderr: error instanceof Error ? error.message : String(error),
            bytes: 0,
          });
          continue;
        }
        rmSync(file, { force: true });
        const profileDir = mkdtempSync(join(tmpdir(), "qpk-visual-qa-"));
        const result = spawnSync(
          chrome,
          [
            "--headless=new",
            "--disable-background-networking",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-gpu",
            "--no-first-run",
            "--hide-scrollbars",
            "--run-all-compositor-stages-before-draw",
            "--force-device-scale-factor=1",
            "--disk-cache-size=0",
            "--media-cache-size=0",
            `--user-data-dir=${profileDir}`,
            `--window-size=${viewport.width},${viewport.height}`,
            "--virtual-time-budget=1000",
            `--screenshot=${file}`,
            url,
          ],
          { encoding: "utf-8", timeout: SCREENSHOT_TIMEOUT_MS },
        );
        rmSync(profileDir, { recursive: true, force: true });
        const bytes = waitForFileBytes(file);
        const exists = existsSync(file);
        if (!exists || bytes < 5_000) {
          report.warnings.push({ route: route.path, viewport: viewport.name, status: result.status, stderr: result.stderr, bytes });
        }
        report.captures.push({ route: route.path, viewport: viewport.name, url, file, bytes, status: result.status });
      }
    }
  } finally {
    stopProcessTree(server);
    writeFileSync(join(OUT_DIR, "server.log"), serverLog.join(""), "utf-8");
    writeFileSync(join(OUT_DIR, "visual-qa-report.json"), JSON.stringify(report, null, 2), "utf-8");
  }

  if (report.warnings.length) {
    console.error(`Visual QA completed with ${report.warnings.length} capture warnings. See ${join(OUT_DIR, "visual-qa-report.json")}`);
    process.exitCode = 1;
  } else {
    console.log(`Visual QA captured ${report.captures.length} screenshots in ${OUT_DIR}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});




