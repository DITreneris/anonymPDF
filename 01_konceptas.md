Koncepcija: AnonymPDF MVP su SQLite

1. Projekto tikslas

Sukurti lengvai diegiamą ir naudojamą Python pagrindu veikiantį įrankį, leidžiantį draudimo bendrovėms automatiškai nuasmeninti PDF dokumentus (pašalinti pavardes, vardus, asmens kodus, kontaktus ir kt.) prieš siunčiant juos į DI analizės platformas.

2. Problemos aprašymas

Draudimo bendrovės susiduria su dideliu kiekvieną dieną generuojamų konfidencialių dokumentų srautu. Norint naudoti tekstų analizės arba DI paslaugas, būtina užtikrinti, kad asmens duomenys būtų pašalinti, laikantis GDPR ir kitų teisės aktų reikalavimų.

3. Vartotojai

Vartotojų skaičius: 25–40 ne techninių skyrių darbuotojų (analitikai, teisės, rizikų valdymo skyrius).

Techninės žinios: pagrindinės darbo su dokumentais kompiuteryje.

4. Sprendimo apžvalga

AnonymPDF – web pagrindu veikiantis SaaS įrankis, kurį kasdien naudoja draudimo bendrovės darbuotojai. Pagrindinės funkcijos:

PDF dokumentų įkėlimas vienu mygtuko paspaudimu

Automatinė teksto ištraukimas ir NER (vardų, asmens kodų, kontaktų) atpažinimas

Asmens duomenų pakeitimas neutraliu ženklu (pvz., [REDACTED])

Anoniminio PDF atsisiuntimas

Audit žurnalas: kas, kada, koks dokumentas

5. Technologijų rinkinys

Front-end: React.js + TypeScript

Back-end: FastAPI (Python) + Uvicorn

Anonymizacija: PyPDF2 + pdfminer.six + spaCy (NER)

Duomenų bazė: SQLite (vietoje PostgreSQL MVP etape) via SQLAlchemy

Failų saugykla: Amazon S3 arba vietinis volume (MVP lygyje .db + failai viename konteineryje)

Autentifikacija: OAuth2/OpenID Connect per Keycloak

Diegimas: Docker + Kubernetes (development: Docker Compose)

6. UI/UX

UI:

Prisijungimo ekranas su įmonės logotipu

Drag-and-drop arba failų pasirinkimo įkėlimo sritis

Progreso juosta ir statuso atnaujinimai

Rezultatų ekranas su miniatiūromis ir atsisiuntimo mygtukais

Audit log lentelė

UX:

Intuityvus srautas: Prisijungti → Įkelti dokumentą → Stebėti progresą → Atsisiųsti

Mobilus ir desktop pritaikymas (responsyvumas)

WCAG 2.1 AA prieinamumo standartai (klaviatūros navigacija, alt tekstai, ARIA etc.)

7. Veikimo charakteristikos

Įkėlimo laikas: ≤3 s už PDF iki 10 MB

Anonimizacijos laikas: ~5–10 s už PDF iki 10 MB

Laikini failai: S3 bucket arba konteinerio volume, gyvavimo laikas iki 24 val.

SQLite: .db vieno failo saugykla, zero-config, pero formos rašymo užraktai (vienalaikės rašymo transakcijos)

8. Naudojamumo metrikos

Prioritetas

Funkcija

KPI

1

Įkėlimas + anonimizacija

Vid. apdorojimo laikas <10 s

2

Patikimumas

Uptime ≥99,5 %

3

Audit log

Klaidų rodiklis <1 %

4

Prieinamumas

WCAG 2.1 AA sėkmė 100 %

5

Skalavimas MVP lygyje

Vienalaikiai rašymai iki 5 vartotojų

9. Laiko juosta (≈12 savaičių)

Paruošiamasis PoC (1 savaitė)

UX/UI dizainas (2 savaitės)

Back-end + SQLite (3 savaitės)

Front-end (3 savaitės)

Integracija ir testavimas (2 savaitės)

Deployment & monitoring (1 savaitė)

10. Išvados

MVP su SQLite užtikrins greitą pradžią ir mažas kaštus. Vėliau, augant naudotojų skaičiui ar duomenų apimčiai, galima pereiti prie PostgreSQL ar kitos serverinės DB, perjungiant DATABASE_URL be didelių pakeitimų.