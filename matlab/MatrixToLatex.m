function [latex] = MatrixToLatex(matrix)
    latex = "\begin{pmatrix}" + newline;
    for i = 1:length(matrix)
        for j = 1:length(matrix(1,:))
            txt = string(matrix(i,j));
            txt = TransformBracketFunctions(txt);
            txt = ConvertWaveFunction(txt);
            txt = replace(txt,"*","");
            latex = latex + txt;
            if j < length(matrix(1,:))
                latex = latex + " & ";
            end
        end
        latex = latex + "\\" + newline;
    end
    latex = latex + "\end{pmatrix}";
end


function [out] = TransformBracketFunctions(text)
    out = text;
    expression = "([a-zA-Z]+_[0-9]+\/[0-9]+)";
    functions = regexp(text,expression,'match');
    for i = 1:length(functions)
        frac = ToFractal(functions(i));
        out = replace(out, "("+functions(i)+")", frac);
    end
end

function [out] = ConvertWaveFunction(text)
    out = insertBefore(text, "cos", "\");
    out = insertBefore(out, "sin", "\");
    out = insertBefore(out, "tan", "\");   
end

function [out] = ToFractal(text)
    txt = split(text,"/");
    out = "\frac{";
    out = out + ConvertToMathSymbol(txt(1)) + "}{" + txt(2) + "}";
end

function [out] = ConvertToMathSymbol(text)
    out = insertBefore(text, "alpha", "\");
    out = insertBefore(out, "omega", "\");
    out = insertAfter(out, "_", "{");
    out = out + "}";
end