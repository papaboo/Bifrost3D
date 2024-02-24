// Directional-hemispherical reflectance for OrenNayar.
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

const int oren_nayar_sample_count = 4096u;
const int oren_nayar_angle_sample_count = 32u;
const int oren_nayar_roughness_sample_count = 32u;

const float oren_nayar[] = {
    // Roughness 0
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    // Roughness 0.0322581
    1.00121f, 1.00095f, 1.00089f, 1.00083f, 1.00077f, 1.00071f, 1.00064f, 1.00058f, 1.00051f, 1.00044f, 1.00037f, 1.0003f, 1.00022f, 1.00015f, 1.00007f, 1.0f, 0.999924f, 0.999848f, 0.999771f, 0.999694f, 0.999616f, 0.999537f, 0.999458f, 0.999378f, 0.999297f, 0.999214f, 0.999129f, 0.99904f, 0.998945f, 0.99884f, 0.998712f, 0.998428f, 
    // Roughness 0.0645161
    1.00454f, 1.00352f, 1.0033f, 1.00307f, 1.00283f, 1.00259f, 1.00234f, 1.00208f, 1.00181f, 1.00154f, 1.00127f, 1.00099f, 1.00071f, 1.00043f, 1.00014f, 0.999849f, 0.999556f, 0.999261f, 0.998964f, 0.998665f, 0.998364f, 0.998061f, 0.997755f, 0.997445f, 0.997131f, 0.996811f, 0.996481f, 0.996138f, 0.995771f, 0.995365f, 0.994871f, 0.993772f, 
    // Roughness 0.0967742
    1.00916f, 1.00698f, 1.00651f, 1.00602f, 1.00551f, 1.005f, 1.00446f, 1.00391f, 1.00335f, 1.00277f, 1.00219f, 1.0016f, 1.001f, 1.00039f, 0.99978f, 0.99916f, 0.998536f, 0.997906f, 0.997273f, 0.996635f, 0.995993f, 0.995346f, 0.994693f, 0.994034f, 0.993364f, 0.992681f, 0.991979f, 0.991246f, 0.990465f, 0.989598f, 0.988545f, 0.986202f, 
    // Roughness 0.129032
    1.01401f, 1.0104f, 1.00962f, 1.00881f, 1.00797f, 1.00712f, 1.00623f, 1.00531f, 1.00438f, 1.00344f, 1.00247f, 1.00149f, 1.0005f, 0.999491f, 0.998475f, 0.997449f, 0.996414f, 0.995372f, 0.994323f, 0.993266f, 0.992203f, 0.991131f, 0.990051f, 0.988958f, 0.987849f, 0.986718f, 0.985554f, 0.98434f, 0.983047f, 0.981611f, 0.979866f, 0.975985f, 
    // Roughness 0.16129
    1.01808f, 1.0129f, 1.01178f, 1.01061f, 1.00941f, 1.00818f, 1.0069f, 1.00559f, 1.00425f, 1.00289f, 1.00151f, 1.0001f, 0.998672f, 0.997227f, 0.995767f, 0.994294f, 0.992808f, 0.991311f, 0.989803f, 0.988286f, 0.986759f, 0.98522f, 0.983667f, 0.982097f, 0.980505f, 0.97888f, 0.977208f, 0.975465f, 0.973607f, 0.971544f, 0.969038f, 0.963464f, 
    // Roughness 0.193548
    1.02061f, 1.01382f, 1.01236f, 1.01082f, 1.00924f, 1.00763f, 1.00596f, 1.00424f, 1.00249f, 1.00071f, 0.99889f, 0.997044f, 0.995174f, 0.99328f, 0.991367f, 0.989435f, 0.987488f, 0.985525f, 0.98355f, 0.981561f, 0.979559f, 0.977542f, 0.975507f, 0.973449f, 0.971362f, 0.969232f, 0.967041f, 0.964756f, 0.962321f, 0.959618f, 0.956333f, 0.949027f, 
    // Roughness 0.225806
    1.02117f, 1.01281f, 1.01101f, 1.00912f, 1.00718f, 1.0052f, 1.00314f, 1.00103f, 0.998872f, 0.996677f, 0.994441f, 0.992169f, 0.989868f, 0.987538f, 0.985183f, 0.982806f, 0.98041f, 0.977995f, 0.975564f, 0.973117f, 0.970653f, 0.968171f, 0.965667f, 0.963136f, 0.960567f, 0.957947f, 0.95525f, 0.952439f, 0.949442f, 0.946115f, 0.942074f, 0.933084f, 
    // Roughness 0.258065
    1.01962f, 1.0098f, 1.00768f, 1.00545f, 1.00317f, 1.00084f, 0.998424f, 0.995935f, 0.993401f, 0.99082f, 0.988191f, 0.98552f, 0.982814f, 0.980073f, 0.977304f, 0.974509f, 0.971691f, 0.968852f, 0.965993f, 0.963115f, 0.960218f, 0.9573f, 0.954355f, 0.951378f, 0.948358f, 0.945276f, 0.942105f, 0.938799f, 0.935275f, 0.931363f, 0.926611f, 0.916039f, 
    // Roughness 0.290323
    1.01606f, 1.00489f, 1.00248f, 0.999946f, 0.997357f, 0.994709f, 0.991959f, 0.989129f, 0.986248f, 0.983313f, 0.980323f, 0.977285f, 0.974208f, 0.971092f, 0.967943f, 0.964765f, 0.96156f, 0.958332f, 0.955081f, 0.951808f, 0.948514f, 0.945194f, 0.941846f, 0.938461f, 0.935026f, 0.931522f, 0.927916f, 0.924156f, 0.920149f, 0.915701f, 0.910296f, 0.898274f, 
    // Roughness 0.322581
    1.01074f, 0.998351f, 0.995677f, 0.992866f, 0.989995f, 0.987059f, 0.98401f, 0.980872f, 0.977677f, 0.974423f, 0.971108f, 0.96774f, 0.964328f, 0.960872f, 0.957381f, 0.953857f, 0.950304f, 0.946724f, 0.943119f, 0.93949f, 0.935838f, 0.932158f, 0.928445f, 0.924691f, 0.920883f, 0.916997f, 0.912999f, 0.908831f, 0.904387f, 0.899455f, 0.893463f, 0.880133f, 
    // Roughness 0.354839
    1.00395f, 0.990479f, 0.987572f, 0.984514f, 0.981392f, 0.978199f, 0.974883f, 0.97147f, 0.967996f, 0.964456f, 0.960851f, 0.957188f, 0.953478f, 0.94972f, 0.945923f, 0.94209f, 0.938226f, 0.934333f, 0.930413f, 0.926466f, 0.922494f, 0.918491f, 0.914454f, 0.910372f, 0.90623f, 0.902004f, 0.897656f, 0.893123f, 0.88829f, 0.882926f, 0.87641f, 0.861913f, 
    // Roughness 0.387097
    0.996028f, 0.981598f, 0.978483f, 0.975207f, 0.971862f, 0.968442f, 0.964889f, 0.961233f, 0.957511f, 0.953719f, 0.949857f, 0.945933f, 0.941957f, 0.937931f, 0.933864f, 0.929758f, 0.925618f, 0.921447f, 0.917247f, 0.913019f, 0.908764f, 0.904476f, 0.90015f, 0.895777f, 0.891339f, 0.886812f, 0.882154f, 0.877298f, 0.87212f, 0.866374f, 0.859393f, 0.843862f, 
    // Roughness 0.419355
    0.987288f, 0.972009f, 0.968711f, 0.965243f, 0.961702f, 0.95808f, 0.954319f, 0.950448f, 0.946507f, 0.942492f, 0.938403f, 0.934248f, 0.930039f, 0.925776f, 0.92147f, 0.917122f, 0.912739f, 0.908323f, 0.903877f, 0.8994f, 0.894894f, 0.890354f, 0.885775f, 0.881144f, 0.876446f, 0.871653f, 0.866721f, 0.861579f, 0.856097f, 0.850013f, 0.842622f, 0.826178f, 
    // Roughness 0.451613
    0.978005f, 0.96198f, 0.958521f, 0.954883f, 0.951168f, 0.947369f, 0.943424f, 0.939364f, 0.93523f, 0.931019f, 0.92673f, 0.922371f, 0.917956f, 0.913485f, 0.908968f, 0.904408f, 0.899811f, 0.895178f, 0.890514f, 0.885819f, 0.881093f, 0.876331f, 0.871527f, 0.86667f, 0.861742f, 0.856714f, 0.851542f, 0.846148f, 0.840398f, 0.834016f, 0.826263f, 0.809015f, 
    // Roughness 0.483871
    0.968418f, 0.951734f, 0.948133f, 0.944345f, 0.940478f, 0.936523f, 0.932416f, 0.928189f, 0.923885f, 0.919501f, 0.915036f, 0.910498f, 0.905902f, 0.901247f, 0.896544f, 0.891797f, 0.887011f, 0.882188f, 0.877333f, 0.872444f, 0.867524f, 0.862566f, 0.857565f, 0.852509f, 0.847378f, 0.842144f, 0.836759f, 0.831143f, 0.825157f, 0.818513f, 0.810442f, 0.792485f, 
    // Roughness 0.516129
    0.958719f, 0.941454f, 0.937728f, 0.933809f, 0.929807f, 0.925714f, 0.921464f, 0.91709f, 0.912636f, 0.9081f, 0.903479f, 0.898784f, 0.894028f, 0.889211f, 0.884344f, 0.879432f, 0.874479f, 0.869489f, 0.864465f, 0.859406f, 0.854314f, 0.849184f, 0.844009f, 0.838777f, 0.833468f, 0.828052f, 0.822479f, 0.816668f, 0.810474f, 0.803599f, 0.795246f, 0.776665f, 
    // Roughness 0.548387
    0.949062f, 0.931285f, 0.927447f, 0.923412f, 0.919291f, 0.915077f, 0.910701f, 0.906197f, 0.901611f, 0.89694f, 0.892182f, 0.887347f, 0.88245f, 0.87749f, 0.872479f, 0.86742f, 0.862321f, 0.857182f, 0.852009f, 0.8468f, 0.841557f, 0.836275f, 0.830946f, 0.825558f, 0.820092f, 0.814515f, 0.808776f, 0.802793f, 0.796415f, 0.789336f, 0.780735f, 0.761602f, 
    // Roughness 0.580645
    0.939564f, 0.921333f, 0.917398f, 0.91326f, 0.909034f, 0.904712f, 0.900224f, 0.895605f, 0.890903f, 0.886112f, 0.881233f, 0.876275f, 0.871252f, 0.866166f, 0.861027f, 0.85584f, 0.85061f, 0.84534f, 0.840035f, 0.834693f, 0.829316f, 0.823899f, 0.818434f, 0.812909f, 0.807303f, 0.801584f, 0.795699f, 0.789563f, 0.783022f, 0.775762f, 0.766942f, 0.747321f, 
    // Roughness 0.612903
    0.930313f, 0.91168f, 0.907658f, 0.903428f, 0.899109f, 0.894692f, 0.890105f, 0.885384f, 0.880577f, 0.875681f, 0.870694f, 0.865627f, 0.860494f, 0.855295f, 0.850043f, 0.844741f, 0.839395f, 0.834009f, 0.828587f, 0.823127f, 0.817632f, 0.812095f, 0.80651f, 0.800862f, 0.795133f, 0.789287f, 0.783273f, 0.777001f, 0.770316f, 0.762896f, 0.753881f, 0.733827f, 
    // Roughness 0.645161
    0.92137f, 0.90238f, 0.89828f, 0.893969f, 0.889567f, 0.885065f, 0.88039f, 0.875579f, 0.87068f, 0.865689f, 0.860607f, 0.855442f, 0.85021f, 0.844912f, 0.839558f, 0.834154f, 0.828707f, 0.823217f, 0.81769f, 0.812126f, 0.806525f, 0.800882f, 0.795189f, 0.789434f, 0.783594f, 0.777636f, 0.771506f, 0.765114f, 0.7583f, 0.750738f, 0.74155f, 0.721111f, 
    // Roughness 0.677419
    0.912778f, 0.893468f, 0.8893f, 0.884917f, 0.88044f, 0.875863f, 0.871109f, 0.866217f, 0.861236f, 0.856161f, 0.850993f, 0.845742f, 0.840422f, 0.835034f, 0.829591f, 0.824097f, 0.818557f, 0.812976f, 0.807356f, 0.801698f, 0.796003f, 0.790265f, 0.784476f, 0.778624f, 0.772686f, 0.766628f, 0.760395f, 0.753896f, 0.746968f, 0.739278f, 0.729936f, 0.709153f, 
    // Roughness 0.709677
    0.904562f, 0.884967f, 0.880737f, 0.876289f, 0.871746f, 0.867101f, 0.862277f, 0.857313f, 0.852258f, 0.847108f, 0.841864f, 0.836535f, 0.831136f, 0.825669f, 0.820145f, 0.81457f, 0.808948f, 0.803284f, 0.797581f, 0.79184f, 0.786061f, 0.780238f, 0.774364f, 0.768425f, 0.762399f, 0.756252f, 0.749926f, 0.743331f, 0.7363f, 0.728497f, 0.719017f, 0.697927f, 
    // Roughness 0.741935
    0.896736f, 0.876884f, 0.872599f, 0.868093f, 0.863491f, 0.858785f, 0.853898f, 0.848869f, 0.843748f, 0.838531f, 0.833218f, 0.827819f, 0.82235f, 0.816812f, 0.811216f, 0.805567f, 0.799872f, 0.794134f, 0.788357f, 0.78254f, 0.776686f, 0.770787f, 0.764836f, 0.75882f, 0.752715f, 0.746487f, 0.740079f, 0.733398f, 0.726276f, 0.71837f, 0.708766f, 0.6874f, 
    // Roughness 0.774194
    0.889302f, 0.86922f, 0.864886f, 0.860327f, 0.855672f, 0.850911f, 0.845967f, 0.84088f, 0.835699f, 0.830422f, 0.825048f, 0.819586f, 0.814054f, 0.808451f, 0.80279f, 0.797076f, 0.791315f, 0.785511f, 0.779666f, 0.773782f, 0.76786f, 0.761892f, 0.755873f, 0.749786f, 0.743611f, 0.737311f, 0.730829f, 0.72407f, 0.716865f, 0.708868f, 0.699152f, 0.677539f, 
    // Roughness 0.806452
    0.882258f, 0.861969f, 0.857589f, 0.852983f, 0.84828f, 0.84347f, 0.838475f, 0.833335f, 0.828101f, 0.82277f, 0.817339f, 0.811821f, 0.806232f, 0.800571f, 0.794852f, 0.789078f, 0.783258f, 0.777393f, 0.771489f, 0.765544f, 0.75956f, 0.753531f, 0.747449f, 0.7413f, 0.735061f, 0.728695f, 0.722146f, 0.715317f, 0.708038f, 0.699958f, 0.690142f, 0.668305f, 
    // Roughness 0.83871
    0.875596f, 0.855118f, 0.850698f, 0.84605f, 0.841303f, 0.836449f, 0.831408f, 0.82622f, 0.820937f, 0.815556f, 0.810076f, 0.804507f, 0.798866f, 0.793152f, 0.78738f, 0.781553f, 0.775679f, 0.76976f, 0.763801f, 0.757801f, 0.751762f, 0.745677f, 0.739538f, 0.733333f, 0.727036f, 0.720611f, 0.714002f, 0.707109f, 0.699762f, 0.691608f, 0.681701f, 0.659662f, 
    // Roughness 0.870968
    0.869302f, 0.848655f, 0.844198f, 0.839511f, 0.834724f, 0.82983f, 0.824747f, 0.819516f, 0.814189f, 0.808764f, 0.803238f, 0.797622f, 0.791934f, 0.786174f, 0.780353f, 0.774478f, 0.768555f, 0.762587f, 0.756578f, 0.750528f, 0.744438f, 0.738303f, 0.732114f, 0.725856f, 0.719507f, 0.713029f, 0.706364f, 0.699415f, 0.692007f, 0.683785f, 0.673795f, 0.651573f, 
    // Roughness 0.903226
    0.863363f, 0.842561f, 0.838071f, 0.833349f, 0.828526f, 0.823595f, 0.818474f, 0.813204f, 0.807838f, 0.802371f, 0.796804f, 0.791146f, 0.785416f, 0.779612f, 0.773748f, 0.767828f, 0.761861f, 0.755848f, 0.749794f, 0.743699f, 0.737564f, 0.731382f, 0.725147f, 0.718842f, 0.712445f, 0.705919f, 0.699204f, 0.692203f, 0.684739f, 0.676455f, 0.666391f, 0.644002f, 
    // Roughness 0.935484
    0.857763f, 0.83682f, 0.832299f, 0.827545f, 0.82269f, 0.817725f, 0.81257f, 0.807264f, 0.801861f, 0.796358f, 0.790752f, 0.785056f, 0.779287f, 0.773444f, 0.76754f, 0.761581f, 0.755573f, 0.749519f, 0.743424f, 0.737287f, 0.731111f, 0.724887f, 0.718609f, 0.712262f, 0.705822f, 0.699252f, 0.692491f, 0.685442f, 0.677928f, 0.669588f, 0.659455f, 0.636915f, 
    // Roughness 0.967742
    0.852485f, 0.831413f, 0.826864f, 0.82208f, 0.817196f, 0.8122f, 0.807013f, 0.801675f, 0.796239f, 0.790701f, 0.785062f, 0.779331f, 0.773526f, 0.767646f, 0.761707f, 0.755711f, 0.749666f, 0.743575f, 0.737442f, 0.731268f, 0.725053f, 0.718792f, 0.712475f, 0.706089f, 0.699609f, 0.692998f, 0.686196f, 0.679104f, 0.671543f, 0.663152f, 0.652957f, 0.630278f, 
    // Roughness 1
    0.847511f, 0.826321f, 0.821747f, 0.816936f, 0.812024f, 0.807001f, 0.801784f, 0.796416f, 0.79095f, 0.785382f, 0.77971f, 0.773947f, 0.76811f, 0.762198f, 0.756224f, 0.750195f, 0.744116f, 0.737991f, 0.731824f, 0.725615f, 0.719366f, 0.713069f, 0.706717f, 0.700295f, 0.693779f, 0.687131f, 0.680291f, 0.673159f, 0.665557f, 0.657118f, 0.646866f, 0.62406f, 
};

float sample_oren_nayar(float wo_dot_normal, float roughness) {
    using namespace Bifrost::Math;

    float roughness_coord = roughness * (oren_nayar_roughness_sample_count - 1);
    int lower_roughness_row = int(roughness_coord);
    int upper_roughness_row = min(lower_roughness_row + 1, oren_nayar_roughness_sample_count - 1);

    float wo_dot_normal_coord = wo_dot_normal * (oren_nayar_angle_sample_count - 1);
    int lower_wo_dot_normal_column = int(wo_dot_normal_coord);
    int upper_wo_dot_normal_column = min(lower_wo_dot_normal_column + 1, oren_nayar_angle_sample_count - 1);

    // Interpolate by wo_dot_normal
    float wo_dot_normal_t = wo_dot_normal * (oren_nayar_angle_sample_count - 1) - lower_wo_dot_normal_column;
    const float* lower_rho_row = oren_nayar + lower_roughness_row * oren_nayar_roughness_sample_count;
    float lower_rho = lerp(lower_rho_row[lower_wo_dot_normal_column], lower_rho_row[upper_wo_dot_normal_column], wo_dot_normal_t);

    const float* upper_rho_row = oren_nayar + upper_roughness_row * oren_nayar_roughness_sample_count;
    float upper_rho = lerp(upper_rho_row[lower_wo_dot_normal_column], upper_rho_row[upper_wo_dot_normal_column], wo_dot_normal_t);

    // Interpolate by roughness
    float roughness_t = roughness_coord - lower_roughness_row;
    return lerp(lower_rho, upper_rho, roughness_t);
}

} // NS Bifrost::Assets::Shading::Rho
