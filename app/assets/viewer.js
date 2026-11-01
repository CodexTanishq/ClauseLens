document.addEventListener('DOMContentLoaded', () => {
  const cards = document.querySelectorAll('.finding-card');
  const marks = document.querySelectorAll('mark.cl-highlight');
  const contractPane = document.querySelector('.contract-pane');
  const findingsPane = document.querySelector('.findings-pane');

  function clearActive() {
    cards.forEach(c => c.classList.remove('active'));
    marks.forEach(m => m.classList.remove('cl-pulse'));
  }

  // Click finding card -> scroll to and pulse highlight
  cards.forEach(card => {
    card.addEventListener('click', () => {
      const findingId = card.getAttribute('data-finding-id');
      clearActive();
      card.classList.add('active');

      const targetMark = document.querySelector(`mark[data-finding-id="${findingId}"]`);
      if (targetMark && contractPane) {
        targetMark.scrollIntoView({ behavior: 'smooth', block: 'center' });
        targetMark.classList.add('cl-pulse');
        setTimeout(() => targetMark.classList.remove('cl-pulse'), 1500);
      }
    });
  });

  // Click highlight in text -> scroll to and select finding card
  marks.forEach(mark => {
    mark.addEventListener('click', () => {
      const findingId = mark.getAttribute('data-finding-id');
      clearActive();
      mark.classList.add('cl-pulse');

      const targetCard = document.querySelector(`.finding-card[data-finding-id="${findingId}"]`);
      if (targetCard && findingsPane) {
        targetCard.classList.add('active');
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  });
});

