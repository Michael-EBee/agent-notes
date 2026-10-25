(() => {
  document.documentElement.classList.add('js');
  const menu = document.querySelector('.menu-toggle');
  const navigation = document.querySelector('#navigation');
  if (!menu || !navigation) return;
  const mobile = window.matchMedia('(max-width: 560px)');
  const sync = () => {
    menu.hidden = !mobile.matches;
    navigation.hidden = mobile.matches;
    menu.setAttribute('aria-expanded', 'false');
  };
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') === 'true';
    navigation.hidden = open;
    menu.setAttribute('aria-expanded', String(!open));
  });
  navigation.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && mobile.matches) {
      navigation.hidden = true;
      menu.setAttribute('aria-expanded', 'false');
      menu.focus();
    }
  });
  mobile.addEventListener('change', sync);
  sync();
})();
