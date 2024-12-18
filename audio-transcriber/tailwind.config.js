/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      // Add custom breakpoints to match our responsive design
      screens: {
        'sm': '640px',
        'md': '768px',
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
      },
      // Add z-index utilities
      zIndex: {
        '40': '40',
        '50': '50',
        '60': '60',
      },
      // Add transition utilities
      transitionProperty: {
        'width': 'width',
        'height': 'height',
        'spacing': 'margin, padding',
      }
    },
  },
  plugins: [
    require('daisyui'),
  ],
  // DaisyUI config
  daisyui: {
    themes: [
      {
        light: {
          ...require("daisyui/src/theming/themes")["light"],
          primary: "#1E40AF",
          secondary: "#9333EA",
          accent: "#FBBF24",
          neutral: "#2A2E37",
          "base-100": "#FFFFFF",
          info: "#3ABFF8",
          success: "#36D399",
          warning: "#FBBD23",
          error: "#F87272",
        },
      },
      {
        dark: {
          ...require("daisyui/src/theming/themes")["dark"],
          primary: "#1E40AF",
          secondary: "#9333EA",
          accent: "#FBBF24",
        },
      },
      "cupcake",
    ],
    darkTheme: "dark",
    base: true,
    styled: true,
    utils: true,
    prefix: "",
    logs: true,
    themeRoot: ":root",
  },
  // Important to prevent conflicts
  important: true,
  // Enable all Tailwind features
  corePlugins: {
    preflight: true,
  },
  // Add custom variants
  variants: {
    extend: {
      backgroundColor: ['active', 'group-hover'],
      textColor: ['group-hover'],
      transform: ['hover', 'focus'],
      translate: ['responsive', 'hover', 'focus'],
    }
  },
}