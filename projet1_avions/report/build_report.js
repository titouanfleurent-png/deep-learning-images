// Génère le rapport et la fiche d'oral du projet 1. Usage : node build_report.js (depuis ce dossier)
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, LevelFormat, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, Footer, PageNumber, ImageRun } = require(path.join(__dirname, '../../_journal/node_modules/docx'));

const ACCENT = '1F4E79', W = 9026, FIG = path.join(__dirname, '../results/figures');
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const p = t => new Paragraph({ spacing: { after: 120 }, children: [new TextRun(t)] });
const li = t => new Paragraph({ numbering: { reference: 'bul', level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] });
const cell = (t, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA },
  shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: 'auto' } : undefined,
  margins: { top: 50, bottom: 50, left: 90, right: 90 },
  children: [new Paragraph({ alignment: o.center ? AlignmentType.CENTER : undefined, children: [new TextRun({ text: String(t), bold: !!o.bold, color: o.color, size: 19 })] })] });
const table = (widths, head, rows) => new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [
  new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, widths[i], { bold: true, fill: ACCENT, color: 'FFFFFF', center: i > 0 })) }),
  ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, widths[i], { center: i > 0, bold: i === 0 })) })),
] });
const caption = t => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 180 }, children: [new TextRun({ text: t, italics: true, size: 18, color: '595959' })] });
const fig = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
  children: [new ImageRun({ type: path.extname(file).slice(1).replace('jpeg', 'jpg'), data: fs.readFileSync(path.join(FIG, file)), transformation: { width: w, height: h } })] });
const title = (t, s) => [new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(t)] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: s, italics: true, color: '595959' })] })];
const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });

const report = [
  ...title("Reconnaître un avion à 3 niveaux", 'Projet 1 du cours de deep learning sur les images. Titouan Fleurent, CNAM, novembre 2026.'),
  h1('En bref'),
  p("Reconnaître la variante exacte d'un avion sur une photo, c'est difficile. Et toutes les erreurs ne se valent pas : confondre un 737-800 avec un 737-900, ce n'est pas très grave, mais le confondre avec un A320, si. Dans ce projet, je regarde si on peut faire moins d'erreurs graves en donnant au réseau la hiérarchie des avions (constructeur, famille, variante)."),
  p("J'ai comparé 3 modèles sur le dataset FGVC-Aircraft, avec 3 entraînements par modèle. Sur le test, les 2 modèles qui utilisent la hiérarchie font 14 à 16 % d'erreurs graves en moins, et ils ne reconnaissent pas moins de variantes. Je ne peux pas les départager sur les erreurs, mais le modèle à soft labels se contredit beaucoup moins entre les niveaux."),
  h1('1. La question'),
  p("Un réseau à qui on donne la hiérarchie des classes fait-il des erreurs moins graves, et est-ce que ça lui coûte des bonnes réponses ? Le projet part de 2 articles : Maji et al. (2013), qui présentent le dataset, et Bertinetto et al. (2020), qui proposent de juger un modèle sur la gravité de ses erreurs, et pas seulement sur leur nombre."),
  h1('2. Les données'),
  li("FGVC-Aircraft : 10 000 photos, 30 constructeurs, 70 familles et 100 variantes (environ 100 images par variante)."),
  li("J'ai gardé les splits officiels : 3 334 images pour l'entraînement, 3 333 pour la validation et 3 333 pour le test. Le modèle ne s'entraîne que sur les 3 334 premières, soit environ 33 images par variante."),
  li("J'enlève d'abord le bandeau de 20 px en bas des photos. Il contient le nom du photographe, et le modèle pourrait s'en servir au lieu de regarder l'avion."),
  li("Je redimensionne les images en carré de 257 px (recadrer directement couperait le nez et la queue), puis je recadre à 224 px : au hasard à l'entraînement, au centre à l'évaluation."),
  li("À l'entraînement, j'ajoute des variations : recadrage au hasard, miroir horizontal, luminosité, contraste et saturation (20 %). Les images sont normalisées comme pour ImageNet, puisque le réseau a été pré-entraîné dessus."),
  h1('3. La méthode'),
  h2('3.1 Les 3 modèles'),
  p("Les 3 modèles ont le même réseau (EfficientNet-B0 pré-entraîné sur ImageNet) et les mêmes réglages. Seule la façon de leur donner la hiérarchie change."),
  table([1900, 2300, 4826], ['Modèle', 'Têtes', 'Fonction de perte'], [
    ['Plat', '1 (100 variantes)', 'Cross-entropy, label smoothing de 0,1'],
    ['3 têtes', '3 (30, 70, 100)', 'CE variante + 0,5 x (CE constructeur + CE famille)'],
    ['Soft labels', '1 (100 variantes)', 'Cross-entropy vers une cible étalée : softmax(-10 x hauteur / 3)'],
  ]), spacer(),
  p("La hauteur mesure la distance entre 2 variantes dans la hiérarchie : 0 si c'est la même variante, 1 si c'est la même famille, 2 si c'est le même constructeur et 3 si les constructeurs sont différents. Avec les soft labels, la bonne variante garde un poids de 0,787 dans la cible, et le reste va surtout aux variantes proches."),
  p("Les modèles à une seule tête ne prédisent que la variante. Pour avoir la famille et le constructeur, j'additionne les probabilités des variantes concernées."),
  h2("3.2 L'entraînement"),
  li("3 epochs avec le réseau pré-entraîné gelé (BatchNorm comprises), pour que seules les têtes apprennent, puis 20 epochs où tout le réseau apprend."),
  li("AdamW, avec un learning rate de 1e-4 pour le réseau et de 1e-3 pour les têtes, un weight decay de 1e-4, une décroissance en cosinus et des batchs de 32."),
  li("Sur MacBook Air M4 (GPU MPS), une epoch prend environ 85 à 105 s. Les 9 entraînements ont pris un peu moins de 5 h."),
  h2("3.3 Comment j'évalue"),
  li("Pour chaque entraînement, je garde le modèle qui a la meilleure accuracy sur les variantes en validation."),
  li("J'ai écrit et figé mon protocole avant de toucher au test (results/protocole_fige.md). Le test n'a été lancé qu'une seule fois."),
  li("Les résultats sont des moyennes et des écarts-types sur 3 graines."),
  li("Pour comparer 2 modèles, j'utilise un bootstrap apparié sur les images (2 000 tirages, intervalle de confiance à 95 %), avec les prédictions moyennées sur les 3 graines."),
  h1('4. Les résultats sur le test'),
  table([3626, 1800, 1800, 1800], ['Métrique', 'Plat', '3 têtes', 'Soft labels'], [
    ['Accuracy variante (%)', '68,7 ± 0,2', '69,6 ± 0,6', '69,6 ± 0,4'],
    ['Accuracy famille (%)', '79,8 ± 0,4', '81,8 ± 0,2', '80,7 ± 0,5'],
    ['Accuracy constructeur (%)', '86,4 ± 0,1', '85,9 ± 0,1', '87,3 ± 0,4'],
    ['Top-5 variante (%)', '89,7 ± 0,3', '90,7 ± 0,2', '91,4 ± 0,1'],
    ['Gravité moyenne, toutes images', '0,676 ± 0,002', '0,622 ± 0,011', '0,626 ± 0,010'],
    ["Gravité moyenne d'une erreur", '2,16 ± 0,01', '2,05 ± 0,01', '2,06 ± 0,01'],
    ['Erreurs graves (% des images)', '14,9 ± 0,2', '12,4 ± 0,3', '12,8 ± 0,3'],
    ['Cohérence des 3 niveaux (%)', '91,9 ± 0,3', '83,7 ± 0,1', '95,4 ± 0,1'],
  ]),
  caption("Tableau 1. Test, moyenne ± écart-type sur 3 graines. Une erreur faite au hasard aurait une gravité de 2,91."),
  fig('test_gravite.png', 470, 211),
  caption("Figure 1. Répartition des images selon la gravité de la prédiction (test, moyenne sur 3 graines)."),
  table([2626, 2400, 2000, 2000], ['Différence', 'Accuracy variante (pts)', 'Gravité moyenne', 'Erreurs graves (pts)'], [
    ['3 têtes - Plat', '+0,95 [0,23 ; 1,60]', '-0,054 [-0,070 ; -0,037]', '-2,44 [-2,99 ; -1,88]'],
    ['Soft - Plat', '+0,89 [0,36 ; 1,45]', '-0,050 [-0,064 ; -0,036]', '-2,13 [-2,66 ; -1,59]'],
    ['Soft - 3 têtes', '-0,06 [-0,77 ; 0,68]', '+0,004 [-0,013 ; 0,020]', '+0,31 [-0,26 ; 0,91]'],
  ]),
  caption("Tableau 2. Différences sur le test, avec intervalle de confiance à 95 % (bootstrap apparié)."),
  h2("Ce que j'en retiens"),
  li("Les 2 modèles qui utilisent la hiérarchie font moins d'erreurs graves que le modèle plat : 16 % de moins pour les 3 têtes et 14 % pour les soft labels. Dans les 2 cas, l'intervalle exclut zéro."),
  li("Ça ne coûte rien sur les variantes, on gagne même environ 1 point. Ce gain reste petit (pour les 3 têtes, l'intervalle est proche de zéro), donc je ne m'appuie pas dessus."),
  li("Je ne peux pas départager les 3 têtes et les soft labels sur les 3 critères testés : tous les intervalles contiennent zéro."),
  li("La vraie différence est ailleurs : le modèle à 3 têtes se contredit sur 16 % des images, parce que ses têtes sont indépendantes. Les soft labels arrivent à 95 % de cohérence avec une seule tête. C'est pour ça que je les ai gardés pour la démo (choix fait sur la validation, avant le test)."),
  li("Le modèle à 3 têtes fait un peu moins bien que le plat sur le constructeur en test (85,9 % contre 86,4 %), alors qu'il faisait mieux en validation (86,4 % contre 85,7 %). Je ne peux donc pas dire qu'il est meilleur à ce niveau."),
  fig('courbes_validation.png', 420, 247),
  caption("Figure 2. Accuracy sur les variantes en validation, epoch par epoch. Le plus gros gain arrive quand on dégèle le réseau."),
  h1("5. Où le modèle se trompe"),
  table([2600, 2600, 1913, 1913], ['Vraie variante', 'Variante prédite', 'Images (sur ~33)', 'Hauteur'], [
    ['C-47', 'DC-3', '15,3', '2'], ['E-190', 'E-195', '14,0', '1'], ['MD-11', 'DC-10', '11,7', '2'], ['DC-3', 'C-47', '11,3', '2'],
    ['A340-200', 'A340-300', '10,7', '1'], ['BAE 146-300', 'BAE 146-200', '10,3', '1'],
  ]),
  caption("Tableau 3. Les confusions les plus fréquentes (soft labels, test, moyenne sur 3 graines)."),
  li("Le modèle confond surtout des avions très proches : des versions plus ou moins longues du même avion, ou des avions dérivés l'un de l'autre (le C-47 est la version militaire du DC-3)."),
  li("La hiérarchie du dataset n'est pas parfaite : le C-47 et le DC-3 sont dans 2 familles différentes, donc cette confusion compte en hauteur 2 alors que les avions se ressemblent énormément."),
  li("Les variantes les plus dures : 737-300 (26 %), 767-300 (33 %), C-47 (36 %), 737-800 (38 %), 747-200 (38 %) et A320 (38 %). Les mieux reconnues ont une forme bien à elles : Il-76, Eurofighter Typhoon et SR-20 (97 %)."),
  li("Les variantes diffèrent souvent par la longueur du fuselage ou le nombre de hublots, ce qui se voit mal à 224 px. C'est une hypothèse : il faudrait refaire un entraînement à plus haute résolution pour la vérifier."),
  h1('6. Où le modèle regarde (Grad-CAM)'),
  fig('gradcam_planche.png', 480, 368),
  caption("Figure 3. Images de validation tirées au hasard. Ligne 1 : bonne variante. Ligne 2 : même famille. Ligne 3 : autre constructeur."),
  p("Grad-CAM montre les zones qui ont le plus compté dans la prédiction. Sur cette planche, pour les bonnes réponses et les erreurs dans la même famille, le modèle regarde presque toujours l'avion. Pour les erreurs graves, il regarde plusieurs fois le sol ou le fond (PA-28, DHC-8). Il a peut-être été gêné par le décor sur ces images."),
  p("Ça reste une observation sur 12 images, pas une preuve. Pour le mesurer proprement, il faudrait des masques qui délimitent les avions."),
  h1('7. La démo'),
  p("J'ai fait une petite application Gradio (app.py) : on met une photo, et elle affiche les 3 meilleures réponses à chaque niveau et la carte Grad-CAM. Testée en local sur un exemple de validation : A380 reconnu à 95 %."),
  fig('demo_gradio.jpg', 240, 232),
  caption("Figure 4. La démo en local."),
  h1('8. Les limites'),
  li("Je n'ai testé qu'une architecture et qu'une résolution. Je n'ai pas fait l'essai prévu avec ConvNeXt-Tiny, parce que ma question porte sur la hiérarchie et pas sur le meilleur réseau."),
  li("Je n'ai pas optimisé w_coarse = 0,5 et beta = 10. D'autres valeurs pourraient changer le classement entre les modèles."),
  li("Le bootstrap porte sur les images : il ne prend pas en compte toute la variabilité entre les entraînements, et 3 graines, c'est peu. Je n'ai pas non plus corrigé pour les comparaisons multiples (9 intervalles)."),
  li("Je n'ai pas testé le modèle sur des photos hors du dataset. Les scores ne valent que pour ce type de photos."),
  li("La gravité dépend de la hiérarchie du dataset, qui ne reflète pas toujours la ressemblance réelle entre les avions."),
  h1('9. Conclusion'),
  p("Sur FGVC-Aircraft, donner la hiérarchie au modèle réduit les erreurs graves de 14 à 16 %, sans perte sur les variantes. Et le plus simple marche très bien : changer uniquement la cible avec des soft labels donne le même gain qu'un modèle à 3 têtes, avec des prédictions plus cohérentes."),
  p("Pour aller plus loin, je testerais une résolution plus élevée pour les variantes proches, je réglerais beta et w_coarse sur la validation, et j'évaluerais le modèle sur des photos hors du dataset."),
  h1('Références'),
  li("Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A. (2013). Fine-Grained Visual Classification of Aircraft. arXiv:1306.5151."),
  li("Bertinetto, L., Mueller, R., Tertikas, K., Samangooei, S., Lord, N. A. (2020). Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks. CVPR."),
  li("Tan, M., Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML."),
  li("Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV."),
];

const qa = (q, a) => [new Paragraph({ spacing: { before: 140, after: 40 }, keepNext: true, children: [new TextRun({ text: q, bold: true })] }), p(a)];
const oral = [
  ...title("Fiche d'oral : projet avions", 'Les choix à savoir justifier, avec une réponse courte pour chacun.'),
  h1('Le message en 30 secondes'),
  p("En donnant au réseau la hiérarchie des avions, on réduit les erreurs graves de 14 à 16 % sans perdre d'accuracy. Changer uniquement les cibles avec des soft labels suffit : même gain qu'un modèle à 3 têtes, avec des prédictions plus cohérentes."),
  h1('Données'),
  ...qa("Pourquoi retirer le bandeau de 20 px ?", "Il contient le nom du photographe. Le modèle pourrait l'utiliser comme raccourci plutôt que de regarder l'avion."),
  ...qa("Pourquoi redimensionner en carré, au risque de déformer ?", "Les photos sont larges. Un recadrage carré couperait le nez et la queue, qui sont discriminants. Une légère déformation coûte moins cher."),
  ...qa("Pourquoi pas de miroir vertical ?", "Un avion n'est presque jamais à l'envers sur ces photos. Le miroir horizontal, lui, reste réaliste."),
  ...qa("Pourquoi la normalisation ImageNet ?", "Le backbone a été pré-entraîné avec ces statistiques, il faut lui fournir des entrées comparables."),
  h1('Modèle et entraînement'),
  ...qa("Pourquoi le transfer learning ?", "Avec environ 33 images d'entraînement par variante, apprendre un CNN de zéro serait voué au surapprentissage. ImageNet apporte déjà des filtres généraux."),
  ...qa("Pourquoi EfficientNet-B0 ?", "Bon compromis entre précision et coût, adapté à un MacBook Air sans ventilateur. La question porte sur la hiérarchie, pas sur le meilleur backbone."),
  ...qa("Pourquoi 3 epochs de gel, BatchNorm comprises ?", "Les têtes partent de valeurs aléatoires : leurs gradients perturberaient un backbone déjà bon. On gèle aussi les BatchNorm, sinon leurs statistiques continueraient à bouger."),
  ...qa("Pourquoi deux taux d'apprentissage ?", "1e-3 pour les têtes, qui ont tout à apprendre, et 1e-4 pour le backbone, qu'on ajuste seulement."),
  ...qa("Comment fonctionnent les soft labels ?", "La cible n'est plus un seul 1. On applique un softmax à -beta x hauteur / 3, ce qui donne du poids aux classes proches. Avec beta = 10, la bonne classe garde 0,787. Plus beta est grand, plus on se rapproche d'une cible classique."),
  ...qa("Pourquoi w_coarse = 0,5 ?", "Pour garder la variante comme objectif principal tout en régularisant avec les niveaux supérieurs. Cette valeur n'a pas été optimisée, c'est une limite."),
  ...qa("Comment comparer à 3 niveaux un modèle à une seule tête ?", "Par marginalisation : la probabilité d'une famille est la somme des probabilités de ses variantes."),
  h1('Évaluation'),
  ...qa("Qu'est-ce que la gravité d'une erreur ?", "La hauteur du plus proche ancêtre commun entre vraie classe et prédiction : 1 même famille, 2 même constructeur, 3 autre constructeur. Une erreur au hasard vaudrait 2,91."),
  ...qa("Pourquoi deux gravités, 2,06 et 0,63 pour le même modèle ?", "2,06 est la moyenne sur les seules erreurs. 0,63 compte les bonnes réponses à 0 : elle tient donc compte du nombre d'erreurs et de leur gravité. Un modèle pourrait améliorer la première en faisant simplement plus d'erreurs bénignes."),
  ...qa("Pourquoi 3 graines et un bootstrap ?", "Les graines mesurent la variabilité de l'entraînement, le bootstrap celle due à l'échantillon d'images. Il est apparié car les modèles sont comparés sur les mêmes images."),
  ...qa("Comment le test a-t-il été protégé ?", "Tous les choix ont été faits sur la validation, le protocole a été écrit avant, et le test n'a été lancé qu'une fois."),
  ...qa("Pourquoi la cohérence du plat n'est-elle pas de 100 % ?", "La famille prédite est celle dont la somme de probabilités est la plus forte. Elle peut différer de la famille de la variante la plus probable, si les probabilités sont réparties entre plusieurs variantes."),
  h1('Questions pièges'),
  ...qa("Vos 70 % ne sont-ils pas faibles ?", "Il y a 100 classes très proches et environ 33 images d'entraînement par classe, à 224 px. Le hasard donnerait 1 %. Une résolution ou un backbone plus élevés amélioreraient sans doute ce score, mais ce n'était pas la question du projet."),
  ...qa("Le gain d'accuracy d'environ 1 point est-il réel ?", "L'intervalle exclut zéro mais reste près de zéro pour le modèle à 3 têtes (0,23 à 1,60). Le résultat solide concerne les erreurs graves, pas l'accuracy."),
  ...qa("Grad-CAM prouve-t-il que le modèle regarde les bons détails ?", "Non. C'est un outil qualitatif, montré sur 12 images tirées au hasard. Il aide à formuler des hypothèses, comme l'influence de l'arrière-plan sur certaines erreurs graves."),
  ...qa("Pourquoi le modèle à 3 têtes n'est-il pas retenu ?", "Il ne fait pas mieux sur les critères testés, se contredit sur 16 % des images et a plus de paramètres."),
  ...qa("Que feriez-vous avec plus de temps ?", "Tester 448 px, régler beta et w_coarse sur la validation, et évaluer sur des photos hors du jeu de données."),
];

function build(children, file, docTitle) {
  const doc = new Document({ creator: 'Titouan Fleurent', title: docTitle,
    styles: { default: { document: { run: { font: 'Calibri', size: 22 } },
      title: { run: { font: 'Calibri', size: 40, bold: true, color: ACCENT }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
      heading1: { run: { font: 'Calibri', size: 28, bold: true, color: ACCENT }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 2 } } } },
      heading2: { run: { font: 'Calibri', size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true } } } },
    numbering: { config: [{ reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } }] }] },
    sections: [{ properties: { page: { margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '808080' })] })] }) }, children }] });
  return Packer.toBuffer(doc).then(b => fs.writeFileSync(path.join(__dirname, file), b));
}
build(report, 'Rapport_projet1_avions.docx', "Reconnaître un avion à 3 niveaux").then(() => build(oral, 'Fiche_oral_projet1.docx', "Fiche d'oral")).then(() => console.log('ok'));
