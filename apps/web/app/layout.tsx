import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "katex/dist/katex.min.css";

import { PwaRegister } from "@/components/pwa-register";

import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist", display: "swap" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono", display: "swap" });

export const metadata: Metadata = {
  title: {
    default: "Quant Portfolio-Kaizen",
    template: "%s | Quant Portfolio-Kaizen",
  },
  description: "Institutional public-data quant research, portfolio construction and paper pre-trade control.",
  manifest: "/manifest.webmanifest",
  icons: { icon: "/qpk-mark.svg" },
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  colorScheme: "dark",
  themeColor: "#070a0f",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable}`}>
      <body>
        <PwaRegister />
        {children}
      </body>
    </html>
  );
}
