const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("qpkDesktop", {
  platform: process.platform,
  runtime: "electron",
});
