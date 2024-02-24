// Directional-hemispherical reflectance for Burley.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------
// Generated by MaterialPrecomputations application.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Math/Utils.h>

namespace Bifrost::Assets::Shading::Rho {

const int burley_sample_count = 4096u;
const int burley_angle_sample_count = 32u;
const int burley_roughness_sample_count = 32u;

const float burley[] = {
    // Roughness 0
    0.503494f, 0.579627f, 0.646261f, 0.70431f, 0.754634f, 0.79803f, 0.835239f, 0.866948f, 0.89379f, 0.916348f, 0.935158f, 0.950706f, 0.963437f, 0.973752f, 0.982013f, 0.988543f, 0.993629f, 0.997525f, 1.00045f, 1.00261f, 1.00415f, 1.00523f, 1.00595f, 1.00641f, 1.00669f, 1.00685f, 1.00693f, 1.00697f, 1.00698f, 1.00698f, 1.00698f, 1.00698f, 
    // Roughness 0.0322581
    0.535527f, 0.60729f, 0.669989f, 0.724516f, 0.771703f, 0.812323f, 0.847092f, 0.87667f, 0.901666f, 0.922637f, 0.940093f, 0.9545f, 0.966276f, 0.975804f, 0.983422f, 0.989436f, 0.994114f, 0.997695f, 1.00038f, 1.00236f, 1.00378f, 1.00477f, 1.00544f, 1.00587f, 1.00613f, 1.00629f, 1.00638f, 1.00642f, 1.00645f, 1.00646f, 1.00648f, 1.00649f, 
    // Roughness 0.0645161
    0.567552f, 0.634945f, 0.693711f, 0.744715f, 0.788766f, 0.826612f, 0.858941f, 0.886389f, 0.909539f, 0.928924f, 0.945028f, 0.958293f, 0.969116f, 0.977856f, 0.984832f, 0.99033f, 0.994602f, 0.997867f, 1.00032f, 1.00212f, 1.00341f, 1.00432f, 1.00493f, 1.00533f, 1.00558f, 1.00574f, 1.00583f, 1.00588f, 1.00592f, 1.00595f, 1.00597f, 1.006f, 
    // Roughness 0.0967742
    0.599568f, 0.662593f, 0.717424f, 0.764907f, 0.805823f, 0.840895f, 0.870785f, 0.896105f, 0.91741f, 0.935208f, 0.949961f, 0.962085f, 0.971955f, 0.979908f, 0.986244f, 0.991227f, 0.995091f, 0.998041f, 1.00025f, 1.00188f, 1.00305f, 1.00387f, 1.00443f, 1.0048f, 1.00503f, 1.00518f, 1.00528f, 1.00534f, 1.00539f, 1.00543f, 1.00547f, 1.00551f, 
    // Roughness 0.129032
    0.631576f, 0.690232f, 0.741131f, 0.785092f, 0.822874f, 0.855172f, 0.882625f, 0.905817f, 0.925278f, 0.941491f, 0.954893f, 0.965877f, 0.974795f, 0.981962f, 0.987656f, 0.992125f, 0.995583f, 0.998218f, 1.00019f, 1.00164f, 1.00269f, 1.00342f, 1.00392f, 1.00426f, 1.00449f, 1.00463f, 1.00473f, 1.00481f, 1.00487f, 1.00492f, 1.00497f, 1.00502f, 
    // Roughness 0.16129
    0.663577f, 0.717864f, 0.76483f, 0.805271f, 0.839919f, 0.869445f, 0.894461f, 0.915525f, 0.933143f, 0.947772f, 0.959824f, 0.969668f, 0.977635f, 0.984016f, 0.98907f, 0.993024f, 0.996076f, 0.998397f, 1.00013f, 1.00141f, 1.00233f, 1.00298f, 1.00343f, 1.00373f, 1.00394f, 1.00409f, 1.00419f, 1.00427f, 1.00434f, 1.00441f, 1.00447f, 1.00453f, 
    // Roughness 0.193548
    0.695569f, 0.745488f, 0.788522f, 0.825443f, 0.856958f, 0.883712f, 0.906292f, 0.92523f, 0.941005f, 0.954051f, 0.964754f, 0.973459f, 0.980475f, 0.986071f, 0.990485f, 0.993925f, 0.996571f, 0.998578f, 1.00008f, 1.00118f, 1.00197f, 1.00253f, 1.00293f, 1.00321f, 1.0034f, 1.00355f, 1.00365f, 1.00374f, 1.00382f, 1.0039f, 1.00398f, 1.00405f, 
    // Roughness 0.225806
    0.727553f, 0.773104f, 0.812206f, 0.845608f, 0.873991f, 0.897974f, 0.918119f, 0.934931f, 0.948865f, 0.960328f, 0.969682f, 0.97725f, 0.983315f, 0.988126f, 0.991901f, 0.994828f, 0.997069f, 0.998761f, 1.00002f, 1.00095f, 1.00162f, 1.0021f, 1.00244f, 1.00269f, 1.00287f, 1.003f, 1.00312f, 1.00322f, 1.00331f, 1.0034f, 1.00348f, 1.00357f, 
    // Roughness 0.258065
    0.75953f, 0.800713f, 0.835884f, 0.865767f, 0.891018f, 0.912232f, 0.929942f, 0.944629f, 0.956722f, 0.966603f, 0.974609f, 0.98104f, 0.986155f, 0.990182f, 0.993318f, 0.995732f, 0.997568f, 0.998946f, 0.999968f, 1.00072f, 1.00126f, 1.00166f, 1.00195f, 1.00216f, 1.00233f, 1.00247f, 1.00258f, 1.00269f, 1.00279f, 1.00289f, 1.00299f, 1.00309f, 
    // Roughness 0.290323
    0.791499f, 0.828314f, 0.859555f, 0.885919f, 0.90804f, 0.926484f, 0.94176f, 0.954324f, 0.964577f, 0.972876f, 0.979536f, 0.984829f, 0.988995f, 0.992239f, 0.994737f, 0.996638f, 0.998069f, 0.999134f, 0.999918f, 1.00049f, 1.00092f, 1.00123f, 1.00146f, 1.00165f, 1.0018f, 1.00193f, 1.00205f, 1.00217f, 1.00228f, 1.00239f, 1.00251f, 1.00262f, 
    // Roughness 0.322581
    0.82346f, 0.855908f, 0.883218f, 0.906065f, 0.925056f, 0.940731f, 0.953575f, 0.964015f, 0.972429f, 0.979148f, 0.984461f, 0.988618f, 0.991836f, 0.994297f, 0.996157f, 0.997546f, 0.998572f, 0.999323f, 0.999871f, 1.00027f, 1.00057f, 1.0008f, 1.00098f, 1.00113f, 1.00127f, 1.0014f, 1.00152f, 1.00165f, 1.00177f, 1.0019f, 1.00202f, 1.00214f, 
    // Roughness 0.354839
    0.855413f, 0.883494f, 0.906874f, 0.926205f, 0.942065f, 0.954973f, 0.965385f, 0.973702f, 0.980278f, 0.985417f, 0.989385f, 0.992407f, 0.994676f, 0.996355f, 0.997577f, 0.998455f, 0.999076f, 0.999515f, 0.999826f, 1.00005f, 1.00022f, 1.00037f, 1.0005f, 1.00062f, 1.00074f, 1.00087f, 1.001f, 1.00113f, 1.00127f, 1.0014f, 1.00154f, 1.00167f, 
    // Roughness 0.387097
    0.887359f, 0.911073f, 0.930524f, 0.946338f, 0.95907f, 0.96921f, 0.977191f, 0.983387f, 0.988124f, 0.991685f, 0.994308f, 0.996195f, 0.997517f, 0.998414f, 0.998999f, 0.999365f, 0.999583f, 0.999709f, 0.999783f, 0.999834f, 0.999883f, 0.999942f, 1.00002f, 1.00011f, 1.00022f, 1.00034f, 1.00047f, 1.00062f, 1.00076f, 1.00091f, 1.00106f, 1.0012f, 
    // Roughness 0.419355
    0.919297f, 0.938644f, 0.954167f, 0.966465f, 0.976068f, 0.983443f, 0.988992f, 0.993068f, 0.995969f, 0.997951f, 0.999229f, 0.999983f, 1.00036f, 1.00047f, 1.00042f, 1.00028f, 1.00009f, 0.999905f, 0.999742f, 0.99962f, 0.999545f, 0.999519f, 0.99954f, 0.999601f, 0.999695f, 0.999815f, 0.999953f, 1.0001f, 1.00026f, 1.00042f, 1.00058f, 1.00074f, 
    // Roughness 0.451613
    0.951228f, 0.966209f, 0.977802f, 0.986585f, 0.993061f, 0.99767f, 1.00079f, 1.00275f, 1.00381f, 1.00422f, 1.00415f, 1.00377f, 1.0032f, 1.00253f, 1.00185f, 1.00119f, 1.0006f, 1.0001f, 0.999704f, 0.999407f, 0.999209f, 0.999099f, 0.999066f, 0.999096f, 0.999176f, 0.999292f, 0.999434f, 0.999592f, 0.999759f, 0.99993f, 1.0001f, 1.00027f, 
    // Roughness 0.483871
    0.983151f, 0.993766f, 1.00143f, 1.0067f, 1.01005f, 1.01189f, 1.01258f, 1.01242f, 1.01165f, 1.01048f, 1.00907f, 1.00756f, 1.00604f, 1.00459f, 1.00327f, 1.00211f, 1.00111f, 1.0003f, 0.999668f, 0.999197f, 0.998875f, 0.998681f, 0.998594f, 0.998594f, 0.998659f, 0.998772f, 0.998918f, 0.999084f, 0.999262f, 0.999444f, 0.999628f, 0.999812f, 
    // Roughness 0.516129
    1.01507f, 1.02132f, 1.02505f, 1.02681f, 1.02703f, 1.02611f, 1.02437f, 1.02209f, 1.01949f, 1.01674f, 1.01399f, 1.01134f, 1.00888f, 1.00666f, 1.0047f, 1.00302f, 1.00163f, 1.0005f, 0.999634f, 0.99899f, 0.998544f, 0.998266f, 0.998125f, 0.998094f, 0.998145f, 0.998255f, 0.998405f, 0.998579f, 0.998767f, 0.998961f, 0.999156f, 0.999352f, 
    // Roughness 0.548387
    1.04698f, 1.04886f, 1.04867f, 1.04691f, 1.04401f, 1.04032f, 1.03616f, 1.03176f, 1.02732f, 1.023f, 1.01891f, 1.01513f, 1.01172f, 1.00872f, 1.00613f, 1.00394f, 1.00214f, 1.00071f, 0.999602f, 0.998784f, 0.998215f, 0.997853f, 0.997659f, 0.997597f, 0.997633f, 0.99774f, 0.997893f, 0.998076f, 0.998274f, 0.998479f, 0.998687f, 0.998894f, 
    // Roughness 0.580645
    1.07888f, 1.07639f, 1.07228f, 1.06701f, 1.06098f, 1.05453f, 1.04794f, 1.04142f, 1.03515f, 1.02925f, 1.02382f, 1.01892f, 1.01457f, 1.01078f, 1.00755f, 1.00486f, 1.00266f, 1.00092f, 0.999573f, 0.998581f, 0.997889f, 0.997443f, 0.997195f, 0.997102f, 0.997124f, 0.997227f, 0.997384f, 0.997575f, 0.997784f, 0.998f, 0.998219f, 0.998438f, 
    // Roughness 0.612903
    1.11077f, 1.10392f, 1.09588f, 1.0871f, 1.07794f, 1.06874f, 1.05972f, 1.05108f, 1.04298f, 1.03551f, 1.02874f, 1.0227f, 1.01741f, 1.01285f, 1.00898f, 1.00578f, 1.00318f, 1.00112f, 0.999546f, 0.998381f, 0.997565f, 0.997035f, 0.996734f, 0.99661f, 0.996617f, 0.996717f, 0.996878f, 0.997077f, 0.997296f, 0.997524f, 0.997754f, 0.997985f, 
    // Roughness 0.645161
    1.14266f, 1.13144f, 1.11948f, 1.10718f, 1.0949f, 1.08293f, 1.07149f, 1.06074f, 1.05081f, 1.04176f, 1.03365f, 1.02649f, 1.02025f, 1.01491f, 1.01041f, 1.0067f, 1.0037f, 1.00133f, 0.999521f, 0.998183f, 0.997243f, 0.99663f, 0.996275f, 0.99612f, 0.996113f, 0.996209f, 0.996374f, 0.996581f, 0.99681f, 0.997049f, 0.997291f, 0.997533f, 
    // Roughness 0.677419
    1.17454f, 1.15896f, 1.14307f, 1.12726f, 1.11186f, 1.09713f, 1.08326f, 1.0704f, 1.05863f, 1.04802f, 1.03857f, 1.03027f, 1.02309f, 1.01697f, 1.01185f, 1.00763f, 1.00422f, 1.00155f, 0.999498f, 0.997987f, 0.996924f, 0.996227f, 0.995819f, 0.995633f, 0.995611f, 0.995704f, 0.995873f, 0.996087f, 0.996327f, 0.996577f, 0.996831f, 0.997084f, 
    // Roughness 0.709677
    1.20641f, 1.18647f, 1.16665f, 1.14733f, 1.12881f, 1.11132f, 1.09503f, 1.08005f, 1.06645f, 1.05427f, 1.04348f, 1.03405f, 1.02593f, 1.01904f, 1.01328f, 1.00855f, 1.00475f, 1.00176f, 0.999477f, 0.997793f, 0.996608f, 0.995827f, 0.995365f, 0.995148f, 0.995112f, 0.995201f, 0.995373f, 0.995596f, 0.995845f, 0.996107f, 0.996372f, 0.996637f, 
    // Roughness 0.741935
    1.23827f, 1.21397f, 1.19022f, 1.16739f, 1.14575f, 1.1255f, 1.10679f, 1.0897f, 1.07427f, 1.06051f, 1.04839f, 1.03784f, 1.02878f, 1.02111f, 1.01471f, 1.00948f, 1.00527f, 1.00198f, 0.999459f, 0.997602f, 0.996293f, 0.995429f, 0.994914f, 0.994666f, 0.994615f, 0.994701f, 0.994877f, 0.995107f, 0.995367f, 0.995639f, 0.995916f, 0.996192f, 
    // Roughness 0.774194
    1.27013f, 1.24146f, 1.21379f, 1.18745f, 1.16269f, 1.13968f, 1.11855f, 1.09934f, 1.08209f, 1.06676f, 1.0533f, 1.04162f, 1.03162f, 1.02317f, 1.01615f, 1.01041f, 1.0058f, 1.00219f, 0.999442f, 0.997413f, 0.995981f, 0.995033f, 0.994465f, 0.994186f, 0.99412f, 0.994202f, 0.994382f, 0.99462f, 0.99489f, 0.995173f, 0.995461f, 0.995749f, 
    // Roughness 0.806452
    1.30198f, 1.26895f, 1.23736f, 1.20751f, 1.17963f, 1.15386f, 1.1303f, 1.10899f, 1.0899f, 1.07301f, 1.05821f, 1.0454f, 1.03446f, 1.02524f, 1.01758f, 1.01134f, 1.00633f, 1.00241f, 0.999428f, 0.997226f, 0.995672f, 0.99464f, 0.994018f, 0.993709f, 0.993628f, 0.993707f, 0.99389f, 0.994136f, 0.994415f, 0.99471f, 0.995009f, 0.995308f, 
    // Roughness 0.83871
    1.33383f, 1.29643f, 1.26092f, 1.22756f, 1.19655f, 1.16803f, 1.14205f, 1.11863f, 1.09772f, 1.07925f, 1.06312f, 1.04918f, 1.0373f, 1.02731f, 1.01902f, 1.01227f, 1.00686f, 1.00264f, 0.999415f, 0.997041f, 0.995364f, 0.994249f, 0.993574f, 0.993234f, 0.993138f, 0.993213f, 0.9934f, 0.993653f, 0.993943f, 0.994248f, 0.994559f, 0.994869f, 
    // Roughness 0.870968
    1.36567f, 1.32391f, 1.28447f, 1.2476f, 1.21348f, 1.1822f, 1.1538f, 1.12826f, 1.10553f, 1.08549f, 1.06802f, 1.05297f, 1.04015f, 1.02938f, 1.02046f, 1.0132f, 1.0074f, 1.00286f, 0.999405f, 0.996859f, 0.995059f, 0.99386f, 0.993132f, 0.992761f, 0.992651f, 0.992722f, 0.992912f, 0.993173f, 0.993473f, 0.993789f, 0.994111f, 0.994432f, 
    // Roughness 0.903226
    1.3975f, 1.35137f, 1.30801f, 1.26764f, 1.2304f, 1.19636f, 1.16554f, 1.1379f, 1.11334f, 1.09173f, 1.07293f, 1.05675f, 1.04299f, 1.03145f, 1.0219f, 1.01413f, 1.00793f, 1.00309f, 0.999397f, 0.996679f, 0.994756f, 0.993474f, 0.992693f, 0.992291f, 0.992166f, 0.992233f, 0.992427f, 0.992696f, 0.993005f, 0.993332f, 0.993665f, 0.993997f, 
    // Roughness 0.935484
    1.42932f, 1.37884f, 1.33155f, 1.28767f, 1.24731f, 1.21052f, 1.17728f, 1.14753f, 1.12114f, 1.09797f, 1.07783f, 1.06053f, 1.04583f, 1.03352f, 1.02334f, 1.01507f, 1.00847f, 1.00331f, 0.999391f, 0.9965f, 0.994456f, 0.99309f, 0.992256f, 0.991823f, 0.991683f, 0.991747f, 0.991943f, 0.99222f, 0.992539f, 0.992877f, 0.993221f, 0.993564f, 
    // Roughness 0.967742
    1.46114f, 1.40629f, 1.35508f, 1.3077f, 1.26422f, 1.22467f, 1.18902f, 1.15715f, 1.12894f, 1.10421f, 1.08274f, 1.06431f, 1.04868f, 1.03559f, 1.02478f, 1.016f, 1.009f, 1.00354f, 0.999387f, 0.996325f, 0.994158f, 0.992709f, 0.991821f, 0.991357f, 0.991203f, 0.991263f, 0.991463f, 0.991747f, 0.992075f, 0.992424f, 0.992779f, 0.993133f, 
    // Roughness 1
    1.49295f, 1.43374f, 1.37861f, 1.32772f, 1.28112f, 1.23882f, 1.20075f, 1.16678f, 1.13674f, 1.11044f, 1.08764f, 1.06809f, 1.05152f, 1.03766f, 1.02622f, 1.01694f, 1.00954f, 1.00377f, 0.999385f, 0.996151f, 0.993862f, 0.99233f, 0.991388f, 0.990894f, 0.990724f, 0.990781f, 0.990984f, 0.991275f, 0.991614f, 0.991973f, 0.992339f, 0.992704f, 
};

float sample_burley(float wo_dot_normal, float roughness) {
    using namespace Bifrost::Math;

    float roughness_coord = roughness * (burley_roughness_sample_count - 1);
    int lower_roughness_row = int(roughness_coord);
    int upper_roughness_row = min(lower_roughness_row + 1, burley_roughness_sample_count - 1);

    float wo_dot_normal_coord = wo_dot_normal * (burley_angle_sample_count - 1);
    int lower_wo_dot_normal_column = int(wo_dot_normal_coord);
    int upper_wo_dot_normal_column = min(lower_wo_dot_normal_column + 1, burley_angle_sample_count - 1);

    // Interpolate by wo_dot_normal
    float wo_dot_normal_t = wo_dot_normal * (burley_angle_sample_count - 1) - lower_wo_dot_normal_column;
    const float* lower_rho_row = burley + lower_roughness_row * burley_roughness_sample_count;
    float lower_rho = lerp(lower_rho_row[lower_wo_dot_normal_column], lower_rho_row[upper_wo_dot_normal_column], wo_dot_normal_t);

    const float* upper_rho_row = burley + upper_roughness_row * burley_roughness_sample_count;
    float upper_rho = lerp(upper_rho_row[lower_wo_dot_normal_column], upper_rho_row[upper_wo_dot_normal_column], wo_dot_normal_t);

    // Interpolate by roughness
    float roughness_t = roughness_coord - lower_roughness_row;
    return lerp(lower_rho, upper_rho, roughness_t);
}

} // NS Bifrost::Assets::Shading::Rho
