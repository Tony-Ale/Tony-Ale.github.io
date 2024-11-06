const themeToggleButton = document.getElementById('theme-toggle');

// Set initial theme based on saved preference or default to dark
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
themeToggleButton.textContent = savedTheme === 'dark' ? 'Light' : 'Dark';

// Toggle theme on button click
themeToggleButton.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Update button text
    themeToggleButton.textContent = newTheme === 'dark' ? 'Light' : 'Dark';
});
