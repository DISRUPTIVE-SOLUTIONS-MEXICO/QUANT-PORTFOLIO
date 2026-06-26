const { app, BrowserWindow, shell } = require("electron");
const path = require("node:path");

const DEFAULT_WEB_URL = "https://quant-portfolio-kaizen-mx.vercel.app";

function allowedOrigin() {
  const configured = process.env.QPK_WEB_URL || DEFAULT_WEB_URL;
  return new URL(configured).origin;
}

function isAllowedInternalUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    return url.origin === allowedOrigin();
  } catch {
    return false;
  }
}

function isAllowedExternalUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    return ["https:", "mailto:"].includes(url.protocol);
  } catch {
    return false;
  }
}

function createWindow() {
  const webUrl = process.env.QPK_WEB_URL || DEFAULT_WEB_URL;
  const window = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1120,
    minHeight: 720,
    backgroundColor: "#070a0f",
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      spellcheck: false,
    },
  });

  window.once("ready-to-show", () => window.show());

  window.webContents.setWindowOpenHandler(({ url }) => {
    if (isAllowedExternalUrl(url)) void shell.openExternal(url);
    return { action: "deny" };
  });

  window.webContents.on("will-navigate", (event, url) => {
    if (!isAllowedInternalUrl(url)) {
      event.preventDefault();
      if (isAllowedExternalUrl(url)) void shell.openExternal(url);
    }
  });

  window.webContents.session.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false);
  });

  void window.loadURL(webUrl);
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
