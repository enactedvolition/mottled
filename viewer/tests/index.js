// `node --test viewer/tests/` resolves the directory as a module (Node >=21
// treats positionals as glob patterns, not directories); this shim makes the
// directory form work. `node --test viewer/tests/parser.test.js` also works.
"use strict";
require("./parser.test.js");
require("./generation.test.js");
require("./features.test.js");
require("./inspector.test.js");
require("./bvh.test.js");
require("./ops.test.js");
require("./models.test.js");
require("./reading.test.js");
require("./blast.test.js");
