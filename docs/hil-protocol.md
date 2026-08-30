# Protokół Hardware-in-the-Loop (HIL)

## Cel i aparatura

HIL weryfikuje deadline'y, synchronizację, interfejsy, awarie łączności i zużycie zasobów
na docelowym komputerze pokładowym. Nie służy do dostrajania modelu. Rejestr aparatury
obejmuje numery seryjne, firmware, kalibracje, schemat połączeń, napięcie zasilania,
wersję autopilota i certyfikat czasu. Kalibracja musi być ważna w chwili każdej próby.

## Bramka wejścia i wyjścia

Wejście: zaliczony SIL, zgodne wersje firmware, skalibrowane czujniki, test E-stop,
izolowana arena oraz podpis operatora bezpieczeństwa. Wyjście: 100 prób bez urazu,
pożaru ani niekontrolowanego uzbrojenia, utrata pakietów p95 <2%, synchronizacja p95
<10 ms, deadline miss <1% i brak P0/P1.

## Obwiednia i człowiek

Pojazd jest unieruchomiony na stanowisku albo w klatce 20 × 20 × 8 m. Dla lotu w klatce:
≤5 m/s poziomo, ≤2 m/s pionowo, przyspieszenie ≤3 m/s², przechył ≤25°, geofence z
buforem 2 m. Pilot z aparaturą w bezpośredniej widoczności ma nadrzędność nad autopilotem;
druga osoba obsługuje fizyczny E-stop, a prowadzący eksperyment nie pełni żadnej z tych ról.

Wstrzykuje się prerejestrowane awarie: opóźnienie i utratę pakietów, bias IMU, drop kamery,
restart procesu oraz ograniczenie mocy obliczeniowej. Profil jest identyczny dla par i nie
przekracza limitów producenta. Telemetria sprzętowa jest mierzona niezależnym loggerem;
zegary synchronizuje się przed blokiem, a dryft kontroluje po nim.

## Procedury awaryjne i raport

Utrata synchronizacji >50 ms, telemetrii >0,5 s, przekroczenie temperatury producenta,
geofence lub oscylacje powodują rozbrojenie/E-stop. Następnie odłączyć energię, odgrodzić
uszkodzony akumulator, udzielić pomocy i zabezpieczyć nośniki tylko gdy jest bezpiecznie.
Raport incydentu zawiera run ID, osoby/role, sprzęt i firmware, dane i model, przebieg,
obrażenia/szkody, logi z hashami, RCA i działania. P0/P1 wymaga zgody kierownika
bezpieczeństwa przed ponownym testem; abort liczy się jako porażka.

Raport końcowy zawiera rozkłady end-to-end latency, jitter, deadline miss, packet loss,
clock offset, temperaturę, moc, energię, RSS/VRAM i degradację względem SIL. Surowe próbki
pozostają dostępne; wartości p95 nie mogą być średnią z percentyli poszczególnych klatek.
