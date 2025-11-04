# ISZ-Project

### HARMONOGRAM
- P1 (08 i 10.10.25) Organizacja Lab i Proj, Grupy dostają różne projekty i minitutoriale dotyczące realizacji różnych modeli/algorytmów

    Zadanie 1,2  Zrealizować model,. zbadać jak się zachowuje, sprawdzić jak działa dla różnych parametrów. Oswoić się z metodologią analizy wyników obliczeń.

- P2  (22-24.10.25) State of the Art (SOTA) – Zadanie 1 - Prezentacje założeń i wyników projektu
- P3  (05-07.11.25) Zadanie 2 - Prezentacje wyników modelowania i analizy rezultatów.
- P4 ( 19-21.11.25) konsultacje projektowe - Zadanie 3

    Zadanie 3   Stworzyć model surogatowy ODE zjawiska, ale tylko czasowy eliminując zmienną przestrzenną. Dobrać ręcznie parametry. Porównać wyniki z oryginalnym modelem czasowo-przestrzennym. Nauczyć na wygenerowanych danych w przedziale uczącym sieć neuronową. Porównać wyniki. Przygotować prezentację.

- P5  (03-05.12.25)   prezentacja „state of research”  (SA)

    Zadanie 4  Na modelu dokonać analizy wrażliwości (SA). Znaleźć zarówno parametry i zmienne dynamiczne które są najbardziej wrażliwe. Dokonać tego dwoma metodami. Dokonać uproszczenia modelu bazując na otrzymanym wyniku. Porównać wyniki. Przygotować prezentację.

    Zadanie 5  Na modelu dokonać asymilacji danych wybranymi dwoma metodami (ABC, 3D-Var). Dane „rzeczywiste” wygenerować z modelu a próbki zaszumić szumem Gaussowskim. Sprawdzić jakość predykcji w przód i w tył na podstawie wybranego fragmentu trajektorii i dla różnej ilości próbek. Można dla kilku fragmentów trajektorii jeżeli bardzo skomplikowana. Podać wyniki dla trzech budżetów czasowych. Przygotować prezentację.

- P6 (17 i 19.12.25)  Konsultacje projektowe  Zadanie 4/5
- P7 ( 07-09.12.24) prezentacje „state of research” 

    Zadanie 6/7 -> Stwórz model surogatowy PINN i sprawdź czy  działa lepiej niż pojedynczy model.

### TEMATYKA PROJEKTÓW
Studenci otrzymują(projekt)/wybierają(laboratoria) model zjawiska opisanego układem równań różniczkowych cząstkowych zależnych od czasu. Implementują ten model.

- Zadanie 1 polega na przygotowaniu prezentacji opisującej model przestrzenno-czasowy PDE, też na podstawie istniejącej literatury. Formułują ostateczny model zjawiska. Konsultują model z czatem AI.

- Zadanie 2 polega na uruchomieniu modelu dla różnych solwerów i porównanie wyników. Wnioski dotyczą (1) czasów obliczeń dla różnych rozdzielczości czasowo-przestrzennych oraz (2) wyboru ekstremalnych parametrów modelu, a także warunków początkowych i brzegowych. (3) Ocenić w jakich przedziałach mogą zawierać się te parametry. (4) Przygotowanie prezentacji.

- Zadanie 3 – Stworzyć model surogatowy ODE zjawiska, ale tylko czasowy eliminując zmienną przestrzenną. Dobrać ręcznie parametry. Porównać wyniki z oryginalnym modelem czasowo-przestrzennym. Przy pomocy danych wygenerowanych w modelu w zadanym przedziale czasowym nauczyć sieć neuronową i porównać wyniki z modelem ODE i PDE. Przygotować prezentację.

- Zadanie 4 Na modelu wejściowym PDE i surogatowym ODE dokonać analizy wrażliwości (SA). Znaleźć zarówno parametry i zmienne dynamiczne, które są najbardziej wrażliwe. Dokonać tego dwoma metodami (Morris, Sobel). Spróbować dokonać uproszczenia modeli bazując na otrzymanym wyniku. Porównać wyniki. Przygotowanie prezentacji.

- Zadanie 5 Na modelach PDE i ODE dokonać asymilacji danych wybranymi dwoma metodami (ABC, 3D-Var). Dane „rzeczywiste” dla modelu PDE wygenerować „ręcznie” a dla modelu ODE wygenerować z modelu PDE do tego dodać dane „rzeczywiste”. Sprawdzić jakość predykcji w przód i w tył na podstawie wybranego fragmentu trajektorii i dla różnej ilości próbek. Można dla kilku fragmentów trajektorii jeżeli bardzo skomplikowana. Podać wyniki dla trzech budżetów czasowych. Przygotować prezentację.

- Zadanie 6 Dla modelu PDE (z zasymilowanymi danymi) stworzyć model PINN i sprawdzić jego wartość predykcyjną w porównaniu z symulacją PDE oraz z modelem ODE. Przygotować prezentację.

- Zadanie 7 Na bazie modelu ODE stworzyć Supermodel, a na bazie sieci PINN z modelu PDE stworzyć SuperNet. Porównać zachowania predykcyjne tych podejść. Przygotować prezentację.