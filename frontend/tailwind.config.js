/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#172b3a",
        muted: "#4b5d6b",
        primary: "#155e63",
        canvas: "#f5f7fa",
        line: "#d6dee5",
      },
      boxShadow: {
        panel: "0 8px 26px rgb(23 43 58 / 6%)",
      },
    },
  },
  plugins: [],
};
