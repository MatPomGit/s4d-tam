# Protokół Software-in-the-Loop (SIL)

## Cel i odpowiedzialność

SIL ocenia zachowanie zamkniętej pętli bez ryzyka fizycznego oraz identyfikuje regresje
przed HIL. Nie zastępuje walidacji sprzętu. Kierownik testu zatwierdza konfigurację,
operator bezpieczeństwa ma wyłączne prawo abortu, a data steward zabezpiecza logi.

## Bramka wejścia i wyjścia

Wejście wymaga poprawnego walidatora ablacji, zamrożonych obrazów kontenerów i danych,
100% testów jednostkowych, deterministycznego odtworzenia oraz braku otwartego incydentu
P0/P1. Wyjście wymaga ukończenia wszystkich scenariuszy, mission success ≥90%, zero
niekontrolowanych naruszeń geofence i collisions/km ≤0,10. Niespełnienie oznacza powrót
do offline, nie zmianę progu po analizie.

## Projekt prób, obwiednia i nadzór

Symulator ogranicza obszar do 500 × 500 m i wysokość AGL 2–120 m; prędkość pozioma
≤15 m/s, pionowa ≤4 m/s, przyspieszenie ≤5 m/s², przechył ≤35° i yaw rate ≤90°/s.
Scenariusze GNSS, wiatr i przeszkody pochodzą wyłącznie z zamrożonej listy. Operator
obserwuje telemetrię i ma niezależny przycisk pause/abort; safety monitor automatycznie
zatrzymuje próbę po geofence, utracie estymacji >1 s lub TTC <0,5 s.

Każdy scenariusz obejmuje nominalny GNSS, stopniową degradację, całkowity zanik, przeszkody
statyczne i dynamiczne oraz jawnie opisany poziom widoczności. Warianty otrzymują identyczny
seed i strumień zakłóceń. Symulator, fizyka, częstotliwości sensorów i krok całkowania są
wersjonowane. Próba zaczyna się po stabilizacji stanu i kończy osiągnięciem celu, timeoutem,
kolizją, geofence lub abortem; wszystkie zakończenia są wynikami, nie brakami danych.

## Awaria i incydent

Po alarmie: zamrozić sterowanie, zapisać stan, zakończyć proces pojazdu, zabezpieczyć logi
i nie wznawiać tego samego run ID. Incydent dostaje ID, czas UTC, wersje, seed, scenariusz,
severity, trigger, działania i hash logów. P0/P1 zamyka bramkę; niezależny reviewer
zatwierdza RCA i test regresji przed wznowieniem. Abort pozostaje nieudaną próbą.

Minimalny zapis obejmuje stan true/estimated, komendy, surowe i dostarczone sensory,
zdarzenia kolizji/geofence/TTC, wykorzystanie zasobów, wersje, seed i zegar monotoniczny.
Raport zbiorczy pokazuje ekspozycję, mianowniki, wszystkie aborty oraz kryteria wyjścia.
