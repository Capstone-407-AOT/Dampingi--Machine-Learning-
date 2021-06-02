const fs = require("fs");

const data = require("./chatette/train/output.json");

// console.log(data.rasa_nlu_data.common_examples);

const datanya = {};
data.rasa_nlu_data.common_examples.forEach((item) => {
  if (datanya[item.intent]) {
    datanya[item.intent].patterns.push(item.text);
  } else {
    datanya[item.intent] = {
      tag: item.intent,
      patterns: [],
    };
  }
});

const datanya_convert = Object.values(datanya);

fs.writeFileSync(
  "./data_training.json",
  JSON.stringify({
    intents: datanya_convert,
  })
);

// console.log(datanya);
