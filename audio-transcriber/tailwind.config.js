/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('daisyui'),
  ],
  daisyui: {
    themes: [
      {
        mytheme: {
          "primary": "#1E40AF",
          "secondary": "#9333EA",
          "accent": "#FBBF24",
          "neutral": "#2A2E37",
          "base-100": "#FFFFFF",
          "info": "#3ABFF8",
          "success": "#36D399",
          "warning": "#FBBD23",
          "error": "#F87272",
        },
      },
       "light", "dark", "cupcake",
      // سایر تم‌ها
    ],
  },
}
