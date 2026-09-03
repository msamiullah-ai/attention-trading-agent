/** @type {import('tailwindcss').Config} */
// NOTE: Tailwind v4 (used here via @tailwindcss/postcss) is CSS-first and
// does not read this file unless imported via `@config` in CSS. The real
// source of truth for theme colors/fonts is the `@theme` block in
// src/styles/global.css. This file is kept only for editor tooling/IDE
// intellisense and mirrors the same values.
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: '#000000',
        surface: '#0A0A0A',
        fg: '#C8FFC8',
        muted: '#6B8F6B',
        border: '#1F2B1F',
        green: '#39FF6A',
        red: '#FF3B3B',
        accent: '#39FF6A',
        danger: '#FF3B3B',
      },
      fontFamily: {
        display: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'IBM Plex Mono', 'Space Mono', 'Courier New'],
      },
    },
  },
  plugins: [],
}
