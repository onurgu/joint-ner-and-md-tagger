/**
 * Created by onur on 7/8/17.
 */

// $('#input_text').onkeyup()

var api_hostname = "localhost";

function htmlEncode(value){
  //create a in-memory div, set it's inner text(which jQuery automatically encodes)
  //then grab the encoded contents back out.  The div never exists on the page.
  return $('<div/>').text(value).html();
}

function htmlDecode(value){
  return $('<div/>').html(value).text();
}

$('#submit_button').bind('click', function() {

      // $("#yourBtnID").hide();

    var elem = $('#input_text');

    // alert(elem[0].value);

    text_area_input = elem[0].value;

    if (text_area_input[text_area_input.length-1] !== "\n") {
        text_area_input += "\n";
    }

    analysis_output_table = $('#analysis_output_table');
    tagger_output_table = $('#tagger_output_table');

    analysis_output_table.find("tbody").replaceWith("<tbody>" +
                                "<tr>" +
                                    "<th>Line no</th>" +
                                    "<th>Surface form</th>" +
                                    "<th>Analyses</th>" +
                                "</tr>" +
                                "</tbody>");
    tagger_output_table.find("tbody").replaceWith("<tbody>" +
                                "<tr>" +
                                    "<th>Line no</th>" +
                                    "<th>Surface form</th>" +
                                    "<th>Predicted NER tag</th>" +
                                "</tr>" +
                                "</tbody>");

    console.log(text_area_input);

    text_area_input_lines = text_area_input.split(/\n/);

    jQuery.post(
        "http://" + api_hostname + ":10000/ner/predict/",
        {'single_line_sentence': text_area_input_lines[0]}
    ).done(
        function (data) {
            console.log(data);
            for (i in data.dataset_file_string) {
                columns = data.dataset_file_string[i]

                tmp_lines = [];

                analyses = columns.slice(2, columns.length-1);
                // for (analysis in analyses) {
                //     is_analysis_correct = "incorrect_analysis";
                //     if (data.disambiguator_output[i - 1] && analyses[analysis] === data.disambiguator_output[i - 1].analysis) {
                //         is_analysis_correct = "correct_analysis"
                //     }
                //     spanned_analysis = "<span class='" + is_analysis_correct + "'>" +
                //         htmlEncode(analyses[analysis]) + "</span>";
                //     tmp_lines.push(spanned_analysis)
                // }

                // htmlEncode(tokens.slice(1).join(" ")).replace(" ", "<br/>")
                analysis_output_table.find("tr:last")
                    .after("<tr> <td>" + i + "</td>" +
                        "<td>" + htmlEncode(columns[0]) + "</td>" +
                        "<td>" + analyses.join("<br/>")  + "</td></tr>")
            }

            for (i in data.tagger_output) {
                columns = data.tagger_output[i];
                tagger_output_table.find("tr:last")
                    .after("<tr> " +
                        "<td>" + i +"</td>" +
                        "<td>" + columns[0] +"</td>" +
                        "<td>" + columns[columns.length-1] +"</td>" +
                        "</tr>"
                    )
            }

        }
    );


});