var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    }
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import Vue from 'vue';
import { EJComponentDecorator } from '@syncfusion/ej2-vue-base';
var StockChartAnnotationsDirective = /** @class */ (function (_super) {
    __extends(StockChartAnnotationsDirective, _super);
    function StockChartAnnotationsDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    StockChartAnnotationsDirective.prototype.render = function () {
        return;
    };
    StockChartAnnotationsDirective = __decorate([
        EJComponentDecorator({})
    ], StockChartAnnotationsDirective);
    return StockChartAnnotationsDirective;
}(Vue));
export { StockChartAnnotationsDirective };
export var StockChartAnnotationsPlugin = {
    name: 'e-stockchart-annotations',
    install: function (Vue) {
        Vue.component(StockChartAnnotationsPlugin.name, StockChartAnnotationsDirective);
    }
};
/**
 * `e-annotation` directive represent a annotation of the VueJS Chart.
 * It must be contained in a Chart component(`ejs-chart`).
 * ```vue
 * <ejs-stockchart>
 *   <e-stockchart-annotations>
 *    <e-annotation content='ID' />
 *    <e-annotation content='ID' />
 *   </e-annotations>
 * </ejs-chart>
 * ```
 */
var StockChartAnnotationDirective = /** @class */ (function (_super) {
    __extends(StockChartAnnotationDirective, _super);
    function StockChartAnnotationDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    StockChartAnnotationDirective.prototype.render = function () {
        return;
    };
    StockChartAnnotationDirective = __decorate([
        EJComponentDecorator({})
    ], StockChartAnnotationDirective);
    return StockChartAnnotationDirective;
}(Vue));
export { StockChartAnnotationDirective };
export var StockChartAnnotationPlugin = {
    name: 'e-stockchart-annotation',
    install: function (Vue) {
        Vue.component(StockChartAnnotationPlugin.name, StockChartAnnotationDirective);
    }
};
