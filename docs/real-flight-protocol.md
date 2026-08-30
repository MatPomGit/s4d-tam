# Protokół prób w locie rzeczywistym

## Cel, status prawny i niezależność bezpieczeństwa

Próba sprawdza wykonalność w ściśle ograniczonej domenie operacyjnej; nie stanowi
certyfikacji ani upoważnienia do autonomicznego lotu. Pilot-in-command (PIC) odpowiada za
lot i może odrzucić plan bez uzasadnienia naukowego. Kierownik badania nie może naciskać na
kontynuację, a operator abortu jest organizacyjnie niezależny od autora algorytmu.

## Bramka i zakres

Wymagane są zaliczone SIL/HIL, ocena ryzyka, zgoda właściciela terenu i właściwego organu,
ważne uprawnienia/ubezpieczenie, NOTAM i pogoda: wiatr stały ≤8 m/s, porywy ≤11 m/s,
widzialność ≥5 km, bez opadu i burzy. Ostateczne limity prawne i instrukcja producenta
mają pierwszeństwo. Wyjście: wykonany plan lub bezpieczny abort, komplet logów/checklist,
zero naruszeń przestrzeni, osób trzecich, P0/P1 i niezgłoszonych incydentów.

## Geofence, dynamika i nadzór

Lot wyłącznie w zatwierdzonym poligonie 300 × 300 m, 30 m od osób niezaangażowanych,
AGL 5–100 m (lub mniej, jeśli wymaga prawo), VLOS, prędkość pozioma ≤12 m/s, pionowa
≤3 m/s, przechył ≤30°. Wirtualny geofence ma bufor 20 m i hard limit. Pilot-in-command
ma aparaturę w ręku i bezwarunkową nadrzędność; niezależny abort operator obserwuje
geofence/TTC, a visual observer przestrzeń. Model nie może rozszerzać obwiedni.

Przed każdym lotem wykonywane są checklisty terenu, pogody, przestrzeni, konstrukcji,
śmigieł, baterii, home position, geofence, return-to-home, łączy, sensorów, czasu, modelu
i E-stop. Najpierw odbywa się lot manualny i kontrolowany test przejęcia. Kolejność wariantów
jest randomizowana dopiero po dopuszczeniu platformy; zmiana baterii, pogody lub pilota jest
rejestrowana jako czynnik blokujący. Osoby i mienie nie są używane jako przeszkody testowe.

## Awaria i zgłaszanie

Przy utracie łącza/GNSS, geofence, TTC <1 s, osobie w strefie lub odchyleniu pogody pilot
przejmuje sterowanie i kolejno: hover (jeśli stabilny), powrót bezpiecznym korytarzem albo
lądowanie w wyznaczonej strefie; przy niestabilności natychmiastowe kontrolowane lądowanie.
Po wypadku: pomoc i służby mają priorytet, następnie zabezpieczenie miejsca, baterii i
logów. PIC dokonuje wymaganych prawem zgłoszeń. W ciągu 24 h rejestr zawiera ID/czas/GPS,
osoby, pogodę, sprzęt, wersje danych/modelu, obrażenia/szkody, trigger, działania i hashe;
RCA zatwierdza komisja niezależna od autora algorytmu. P0/P1 wstrzymuje wszystkie loty.

Każdy lot ma niepowtarzalny run ID oraz zsynchronizowane logi autopilota, komputera,
aparatury i obserwatora. Raportuje się również loty niewystartowane, takeover, utratę łącza,
minimalny clearance, ekspozycję czasową/dystansową i odstępstwa od checklisty. Materiały
z lokalizacją lub wizerunkiem podlegają kontroli dostępu i anonimizacji przed publikacją.
